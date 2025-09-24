import logging  # noqa: I001
import os
from io import BytesIO
from typing import Annotated, Any

from transformers import CLIPModel, CLIPProcessor
import numpy as np
import requests
import structlog
import torch
from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse, Response
from mangum import Mangum
from PIL import Image

from poprox_concepts import Article
from poprox_concepts.api.recommendations.v3 import ProtocolModelV3_0, RecommendationRequestV3, RecommendationResponseV3
from poprox_recommender.api.gzip import GzipRoute
from poprox_recommender.config import default_device
from poprox_recommender.paths import model_file_path
from poprox_recommender.recommenders import load_all_pipelines, select_articles
from poprox_recommender.topics import user_locality_preference, user_topic_preference

logger = logging.getLogger(__name__)

app = FastAPI()


app.router.route_class = GzipRoute

logger = logging.getLogger(__name__)


# Global CLIP model cache
_clip_model = None
_clip_preprocess = None


def get_clip_model():
    """Get cached CLIP model or load it if not cached"""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            device = default_device()
            logger.info(f"Loading CLIP model on device: {device}")
            model_path = model_file_path("openai/clip-vit-base-patch32")  # 768 dimensions

            # Load model components with error handling
            full_model = CLIPModel.from_pretrained(model_path)
            _clip_model = full_model.vision_model.to(device)
            _clip_preprocess = CLIPProcessor.from_pretrained(model_path, use_fast=True)

            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"Failed to load CLIP model: {e}") from e
    return _clip_model, _clip_preprocess


def generate_clip_embedding(image):
    """Generate CLIP embedding for a single image"""
    try:
        model, preprocess = get_clip_model()
        device = default_device()

        # Download and process image
        if not (hasattr(image, "url") and image.url):
            raise ValueError(f"No URL found for image {image.image_id}")

        response = requests.get(image.url, timeout=30)  # Increased timeout
        response.raise_for_status()

        if len(response.content) == 0:
            raise ValueError(f"Empty image content from URL: {image.url}")

        pil_image = Image.open(BytesIO(response.content)).convert("RGB")

        # Validate image dimensions
        if pil_image.size[0] == 0 or pil_image.size[1] == 0:
            raise ValueError(f"Invalid image dimensions: {pil_image.size}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image {image.image_id} from {image.url}: {e}")
        raise ValueError(f"Failed to download image: {e}") from e
    except Exception as e:
        logger.error(f"Failed to process image {image.image_id}: {e}")
        raise ValueError(f"Failed to process image: {e}") from e

    try:
        # Preprocess and encode
        image_tensor = preprocess(images=pil_image, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            image_features = model(image_tensor).last_hidden_state.mean(dim=1)
            # Normalize the features
            norm = image_features.norm(dim=-1, keepdim=True)
            image_features = image_features / (norm + 1e-8)  # Add epsilon to prevent division by zero

        # Convert to list for JSON serialization
        embedding = image_features.cpu().numpy().flatten().tolist()

        # Validate embedding
        if len(embedding) == 0:
            raise ValueError("Generated empty embedding")

        logger.debug(f"Generated embedding of dimension {len(embedding)} for image {image.image_id}")
        return embedding

    except Exception as e:
        logger.error(f"Failed to generate CLIP embedding for image {image.image_id}: {e}")
        raise ValueError(f"Failed to generate CLIP embedding: {e}") from e


@app.get("/warmup")
def warmup(response: Response):
    # Headers set on the response param get included in the response wrapped around return val
    response.headers["poprox-protocol-version"] = ProtocolModelV3_0().protocol_version.value

    # Load and cache available recommenders
    available_recommenders = load_all_pipelines(device=default_device())

    # Load CLIP model during warmup
    try:
        get_clip_model()
        logger.info("CLIP model loaded during warmup")
    except Exception as e:
        logger.warning(f"Failed to load CLIP model during warmup: {e}")

    return list(available_recommenders.keys())


# Article -> dict[UUID, dict[str, np.array]]
@app.post("/embed")
def embed(
    body: Annotated[dict[str, Any], Body()],
    pipeline: str | None = None,
):
    logger.info("Embedding request received")

    article = Article.model_validate(body)
    embeddings = {}

    # Generate image embeddings
    if article.images:
        logger.info(f"Processing {len(article.images)} images for article {article.article_id}")

        # Generate embeddings for each image - let failures propagate
        for image in article.images:
            try:
                embedding_vector = generate_clip_embedding(image)
                embeddings[image.image_id] = {"image": embedding_vector}
                logger.debug(f"Generated embedding for image {image.image_id}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding for image {image.image_id}: {e}")
                # Continue processing other images even if one fails
    else:
        logger.info(f"No images found for article {article.article_id}")

    total_embeddings = len([k for k, v in embeddings.items() if "image" in v])
    logger.info(f"Generated embeddings: {total_embeddings} images")
    return ORJSONResponse(embeddings)


@app.post("/")
def root(
    body: Annotated[dict[str, Any], Body()],
    pipeline: str | None = None,
):
    logger.info(f"Decoded body: {body}")

    req = RecommendationRequestV3.model_validate(body)

    candidate_articles = req.candidates.articles
    num_candidates = len(candidate_articles)

    if num_candidates < req.num_recs:
        msg = f"Received insufficient candidates ({num_candidates}) in a request for {req.num_recs} recommendations."
        raise ValueError(msg)

    logger.info(f"Selecting articles from {num_candidates} candidates...")

    profile = req.interest_profile
    profile.click_topic_counts = user_topic_preference(req.interacted.articles, profile.click_history)
    profile.click_locality_counts = user_locality_preference(req.interacted.articles, profile.click_history)

    embeddings = req.embeddings
    # XXX: If we change the over-the-wire format to numpy instead of list of float we can probably get rid of this.
    for embedding_dict in embeddings.values():
        for key in embedding_dict:
            embedding_dict[key] = np.array(embedding_dict[key], dtype=np.float32)

    outputs = select_articles(
        req.candidates,
        req.interacted,
        profile,
        embeddings,
        {"pipeline": pipeline},
    )

    resp_body = RecommendationResponseV3.model_validate(
        {"recommendations": outputs.default, "recommender": outputs.meta.model_dump()}
    )

    logger.info(f"Response body: {resp_body}")
    return resp_body.model_dump()


handler = Mangum(app)

if "AWS_LAMBDA_FUNCTION_NAME" in os.environ and not structlog.is_configured():
    # Serverless doesn't set up logging like the AWS Lambda runtime does, so we
    # need to configure base logging ourselves. The AWS_LAMBDA_RUNTIME_API
    # environment variable is set in a real runtime environment but not the
    # local Serverless run, so we can check for that.  We will log at DEBUG
    # level for local testing.
    if "AWS_LAMBDA_RUNTIME_API" not in os.environ:
        logging.basicConfig(level=logging.DEBUG)
        # make sure we have debug for all of our code
        logging.getLogger("poprox_recommender").setLevel(logging.DEBUG)
        logger.info("local logging enabled")

    # set up structlog to dump to standard logging
    # TODO: enable JSON logs
    structlog.configure(
        [
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.MaybeTimeStamper(),
            structlog.processors.KeyValueRenderer(key_order=["event", "timestamp"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    structlog.stdlib.get_logger(__name__).info(
        "structured logging initialized",
        function=os.environ["AWS_LAMBDA_FUNCTION_NAME"],
        region=os.environ.get("AWS_REGION", None),
    )
