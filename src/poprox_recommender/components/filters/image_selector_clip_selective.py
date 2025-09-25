import logging
import random
from uuid import UUID

import numpy as np
import torch
from lenskit import Component

from poprox_concepts.domain import CandidateSet, RecommendationList

logger = logging.getLogger(__name__)


class SelectiveImageSelector(Component):
    config: None

    def __call__(
        self,
        recommendations: RecommendationList,
        interacted_articles: CandidateSet,
        embedding_lookup: dict[UUID, dict[str, np.ndarray]],
        **kwargs: object,
    ) -> RecommendationList:
        """Select personalized images for 50% of articles (odd or even positions) using CLIP embeddings."""
        # Generate user embedding from clicked article images
        clip_user_embedding = self._generate_user_embedding(interacted_articles, embedding_lookup)
        if clip_user_embedding is None:
            logger.debug("No user embedding generated, returning original recommendations")
            return recommendations

        # Randomly decide whether to personalize odd or even positioned articles
        personalize_odd = random.choice([True, False])
        logger.debug(f"Personalizing {'odd' if personalize_odd else 'even'} positioned articles")

        logger.debug(
            f"Generated user embedding, personalizing images for ~50% of {len(recommendations.articles)} articles"
        )

        # Initialize extras if not present
        if not recommendations.extras:
            recommendations.extras = [{} for _ in recommendations.articles]

        # Ensure extras list has the same length as articles
        while len(recommendations.extras) < len(recommendations.articles):
            recommendations.extras.append({})

        # Track personalization statistics
        personalized_count = 0
        total_articles_with_images = 0

        # Select best image for each article based on position
        for idx, article in enumerate(recommendations.articles):
            # Initialize extra dict for this article if needed
            if recommendations.extras[idx] is None:
                recommendations.extras[idx] = {}

            extra = recommendations.extras[idx]

            # Check if this article should be personalized based on its position
            is_even_position = (idx % 2) == 0  # 0-indexed, so 0, 2, 4... are even positions
            should_personalize = (personalize_odd and not is_even_position) or (
                not personalize_odd and is_even_position
            )

            # Log personalization decision for all articles (regardless of whether they have images)
            recommendations.extras[idx]["image_personalization"] = {
                "strategy": "selective",
                "attempted": should_personalize,
                "succeeded": article.preview_image_id is not None,
                "personalized": article.preview_image_id is not None,
                "total_images": len(article.images) if article.images else 0,
                "position": idx,
                "original_preview_image_id": article.preview_image_id,
                "new_preview_image_id": article.preview_image_id,
            }

            if not article.images:
                extra["image_personalization"]["personalization_skipped_reason"] = "no_images"
                continue

            total_articles_with_images += 1

            if not should_personalize:
                extra["image_personalization"]["personalization_skipped_reason"] = "position_not_selected"
                continue

            # Get embeddings for all images in this article
            valid_embeddings = []
            valid_images = []

            for img in article.images:
                if img.image_id in embedding_lookup and "image" in embedding_lookup[img.image_id]:
                    embedding_data = embedding_lookup[img.image_id]["image"]
                    embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                    valid_embeddings.append(embedding_tensor)
                    valid_images.append(img)

            # Select best image if we have embeddings
            if valid_embeddings:
                image_embeddings = torch.stack(valid_embeddings)
                best_image = self._select_best_image(image_embeddings, clip_user_embedding, valid_images)
                if best_image:
                    article.preview_image_id = best_image.image_id
                    personalized_count += 1

                    # Update personalization logging
                    extra["image_personalization"]["new_preview_image_id"] = str(best_image.image_id)
                    extra["image_personalization"]["total_images"] = len(article.images)
                else:
                    extra["image_personalization"]["personalization_skipped_reason"] = "no_best_image_selected"
            else:
                extra["image_personalization"]["personalization_skipped_reason"] = "no_valid_embeddings"

        # Log final statistics
        logger.info(
            f"CLIP Selective Personalization Summary: "
            f"Personalized {personalized_count}/{total_articles_with_images} articles with images "
            f"({personalized_count}/{len(recommendations.articles)} total articles). "
            f"Strategy: {'odd' if personalize_odd else 'even'} positions"
        )

        return recommendations

    def _generate_user_embedding(self, interacted_articles: CandidateSet, embedding_lookup) -> torch.Tensor | None:
        """Generate user embedding by averaging CLIP embeddings of preview images."""
        valid_embeddings = []

        for article in interacted_articles.articles[-50:]:  # Use last 50 articles
            image_id_to_use = None

            # Try preview_image_id first (for articles with personalized images)
            if article.preview_image_id and article.preview_image_id in embedding_lookup:
                image_id_to_use = article.preview_image_id
            # Fallback to first image if preview_image_id not available (for older articles)
            elif article.images and len(article.images) > 0 and article.images[0].image_id in embedding_lookup:
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
            logger.debug(f"No valid embeddings found from {len(interacted_articles.articles)} interacted articles")
            return None

        logger.debug(f"Generated user embedding from {len(valid_embeddings)} valid image embeddings")

        # Average and normalize (with epsilon to avoid division by zero)
        user_embedding = torch.mean(torch.stack(valid_embeddings), dim=0)
        user_embedding = user_embedding / (torch.norm(user_embedding) + 1e-8)

        return user_embedding

    def _select_best_image(self, image_embeddings: torch.Tensor, user_embedding: torch.Tensor, images: list):
        """Select the best image using cosine similarity."""
        if len(images) == 0:
            return None

        # Normalize image embeddings (with epsilon to avoid division by zero)
        norms = torch.norm(image_embeddings, dim=1, keepdim=True)
        image_embeddings_norm = image_embeddings / (norms + 1e-8)

        # Compute similarities
        similarities = torch.matmul(image_embeddings_norm, user_embedding)
        best_index = torch.argmax(similarities).item()

        return images[best_index]
