import logging
import random
from dataclasses import dataclass
from uuid import UUID

import numpy as np
import torch
from lenskit import Component

from poprox_concepts.domain import CandidateSet, RecommendationList

logger = logging.getLogger(__name__)


@dataclass
class MultiModalImageSelectorConfig:
    """Configuration for selective multi-modal image selector.
    - Visual user preferences + Text-image relevance
    - visual_weight: How much to weight user's visual taste from click history
    - text_weight: How much to weight image relevance to current article text
    """
    visual_weight: float = 0.7
    text_weight: float = 0.3
    max_history_visual: int = 50


class MultiModalImageSelector(Component):
    config: MultiModalImageSelectorConfig

    def __call__(
        self,
        recommendations: RecommendationList,
        interacted_articles: CandidateSet,
        embedding_lookup: dict[UUID, dict[str, np.ndarray]],
        **kwargs: object,
    ) -> RecommendationList:
        """
        Select personalized images for 50% of articles using multi-modal CLIP embeddings.

        Combines visual user preferences with text-image relevance for better personalization.
        """
        # Generate visual user embedding from clicked article images
        visual_user_embedding = self._generate_visual_user_embedding(interacted_articles, embedding_lookup)

        if visual_user_embedding is None:
            logger.debug("No visual user embedding generated, returning original recommendations")
            return recommendations

        # Randomly decide whether to personalize odd or even positioned articles
        personalize_odd = random.choice([True, False])
        logger.debug(f"Personalizing {'odd' if personalize_odd else 'even'} positioned articles")

        logger.debug(
            f"Generated multi-modal user embedding, personalizing images for ~50% of "
            f"{len(recommendations.articles)} articles"
        )

        # Track selected images for diversity (optional enhancement)
        selected_images = []

        # Select best image for each article based on position
        for idx, article in enumerate(recommendations.articles):
            if not article.images:
                continue

            # Check if this article should be personalized based on its position
            is_even_position = (idx % 2) == 0  # 0-indexed, so 0, 2, 4... are even positions
            should_personalize = (personalize_odd and not is_even_position) or (
                not personalize_odd and is_even_position
            )

            if not should_personalize:
                continue

            # Multi-modal image selection
            best_image = self._select_best_image_multimodal(
                article,
                visual_user_embedding,
                embedding_lookup,
                selected_images
            )

            if best_image:
                article.preview_image_id = best_image.image_id
                selected_images.append(best_image)

        return recommendations

    def _generate_visual_user_embedding(
        self, interacted_articles: CandidateSet, embedding_lookup
    ) -> torch.Tensor | None:
        """Generate user visual preference embedding from clicked article images."""
        valid_embeddings = []

        for article in interacted_articles.articles[-self.config.max_history_visual:]:  # Use configurable history
            image_id_to_use = None

            # Try preview_image_id first (for articles with personalized images)
            if article.preview_image_id and article.preview_image_id in embedding_lookup:
                image_id_to_use = article.preview_image_id
            # Fallback to first image if preview_image_id not available (for older articles)
            elif (
                article.images
                and len(article.images) > 0
                and article.images[0].image_id in embedding_lookup
            ):
                image_id_to_use = article.images[0].image_id

            if (
                image_id_to_use is not None
                and image_id_to_use in embedding_lookup
                and "image" in embedding_lookup[image_id_to_use]
            ):
                embedding_data = embedding_lookup[image_id_to_use]["image"]
                embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                valid_embeddings.append(embedding_tensor)

        if not valid_embeddings:
            logger.debug(
                f"No valid visual embeddings found from {len(interacted_articles.articles)} interacted articles"
            )
            return None

        logger.debug(f"Generated visual user embedding from {len(valid_embeddings)} valid image embeddings")

        # Average and normalize (with epsilon to avoid division by zero)
        stacked_embeddings = torch.stack(valid_embeddings)
        user_embedding = torch.mean(stacked_embeddings, dim=0)
        user_embedding = user_embedding / (torch.norm(user_embedding) + 1e-8)

        # Clean up intermediate tensors to prevent memory accumulation
        del stacked_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return user_embedding

    def _select_best_image_multimodal(
        self,
        article,
        visual_user_emb: torch.Tensor,
        embedding_lookup: dict,
        selected_images: list
    ):
        """Select the best image using multi-modal similarity scores."""

        # Get visual similarity scores
        visual_scores = self._compute_visual_similarities(article, visual_user_emb, embedding_lookup)

        if visual_scores is None:
            return None

        # Get text-image relevance scores (how images relate to CURRENT article text)
        text_relevance_scores = self._compute_text_image_relevance(article, embedding_lookup)

        # Combine scores using weighted fusion
        if text_relevance_scores is not None:
            # Configurable fusion weights
            final_scores = (self.config.visual_weight * visual_scores +
                          self.config.text_weight * text_relevance_scores)
            logger.debug(f"Using multi-modal fusion (visual: {self.config.visual_weight}, "
                        f"text: {self.config.text_weight})")
        else:
            # Fallback to visual-only
            final_scores = visual_scores
            logger.debug("Using visual-only selection (no text embeddings available)")

        # Select best image
        best_idx = torch.argmax(final_scores).item()
        return article.images[best_idx]

    def _compute_visual_similarities(self, article, visual_user_emb: torch.Tensor, embedding_lookup: dict):
        """Compute visual similarity scores between article images and user visual preferences."""
        valid_embeddings = []
        valid_images = []

        for img in article.images:
            if img.image_id in embedding_lookup and "image" in embedding_lookup[img.image_id]:
                embedding_data = embedding_lookup[img.image_id]["image"]
                embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                valid_embeddings.append(embedding_tensor)
                valid_images.append(img)

        if not valid_embeddings:
            return None

        # Stack embeddings and normalize
        image_embeddings = torch.stack(valid_embeddings)
        norms = torch.norm(image_embeddings, dim=1, keepdim=True)
        image_embeddings_norm = image_embeddings / (norms + 1e-8)

        # Compute similarities with user visual preferences
        similarities = torch.matmul(image_embeddings_norm, visual_user_emb)

        # Clean up intermediate tensors
        del image_embeddings, norms, image_embeddings_norm

        return similarities

    def _compute_text_image_relevance(self, article, embedding_lookup: dict):
        """Compute how well each image relates to the article's text content."""

        # Get article text embedding
        if article.article_id not in embedding_lookup or "text" not in embedding_lookup[article.article_id]:
            return None

        article_text_emb = torch.tensor(embedding_lookup[article.article_id]["text"], dtype=torch.float32)
        article_text_emb = article_text_emb / (torch.norm(article_text_emb) + 1e-8)

        # For each image, compute cross-modal similarity with article text
        relevance_scores = []
        for img in article.images:
            if img.image_id in embedding_lookup and "image" in embedding_lookup[img.image_id]:
                img_emb = torch.tensor(embedding_lookup[img.image_id]["image"], dtype=torch.float32)
                img_emb_norm = img_emb / (torch.norm(img_emb) + 1e-8)

                # Cross-modal similarity: how well does this image represent the article's text?
                cross_modal_similarity = torch.dot(img_emb_norm, article_text_emb)
                relevance_scores.append(cross_modal_similarity)
            else:
                relevance_scores.append(torch.tensor(0.0))

        if not relevance_scores:
            return None

        stacked_scores = torch.stack(relevance_scores)

        # Clean up article text embedding after use
        del article_text_emb

        return stacked_scores

    def _select_best_image(self, image_embeddings: torch.Tensor, user_embedding: torch.Tensor, images: list):
        """Legacy method for compatibility - delegates to multi-modal selection."""
        # This method is kept for backward compatibility
        # In practice, _select_best_image_multimodal should be used
        if len(images) == 0:
            return None

        # Normalize image embeddings (with epsilon to avoid division by zero)
        norms = torch.norm(image_embeddings, dim=1, keepdim=True)
        image_embeddings_norm = image_embeddings / (norms + 1e-8)

        # Compute similarities
        similarities = torch.matmul(image_embeddings_norm, user_embedding)
        best_index = torch.argmax(similarities).item()

        return images[best_index]
