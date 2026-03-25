"""
ImageStore: Session-based in-memory storage for images.

Keyed by session UUID. Each session holds up to 4 input ImageModel slots
plus output slots for mixing results.
"""

import uuid
from typing import Optional

from backend.domain.image_model import ImageModel
from backend.domain.resize_policy import ResizeMode, ResizePolicy


class Session:
    """
    Represents a single user session with image slots.

    Attributes:
        id: Session UUID.
        slots: Dict mapping slot index to ImageModel.
        resize_policy: Active resize policy (or None).
    """

    MAX_INPUT_SLOTS = 4
    MAX_OUTPUT_SLOTS = 2
    TOTAL_SLOTS = MAX_INPUT_SLOTS + MAX_OUTPUT_SLOTS  # slots 0-3 = input, 4-5 = output

    def __init__(self, session_id: str):
        self.id = session_id
        self.slots: dict[int, ImageModel] = {}
        self.resize_policy: Optional[ResizePolicy] = None

    def get_image(self, slot: int) -> Optional[ImageModel]:
        """Get the ImageModel at a slot, or None if empty."""
        return self.slots.get(slot)

    def set_image(self, slot: int, image: ImageModel) -> None:
        """Set an ImageModel at a slot."""
        if slot < 0 or slot >= self.TOTAL_SLOTS:
            raise ValueError(
                f"Slot {slot} out of range (0-{self.TOTAL_SLOTS - 1})"
            )
        self.slots[slot] = image

    def get_loaded_input_sizes(self) -> list[tuple[int, int]]:
        """Get (height, width) for all loaded input images."""
        sizes = []
        for i in range(self.MAX_INPUT_SLOTS):
            img = self.slots.get(i)
            if img and img.is_loaded:
                arr = img.get_active_array()
                sizes.append(arr.shape[:2])
        return sizes

    def apply_resize_policy(self, policy: ResizePolicy) -> list[int]:
        """
        Apply a resize policy to all loaded input images.

        Args:
            policy: The resize policy to apply.

        Returns:
            List of slot indices that were resized.
        """
        self.resize_policy = policy
        sizes = self.get_loaded_input_sizes()

        if len(sizes) < 1:
            return []

        # For SMALLEST/LARGEST, need at least 2 images; with 1 image, use its own size
        if len(sizes) == 1 and policy.mode != ResizeMode.FIXED:
            return []

        target = policy.compute_target_size(sizes)
        affected = []

        for i in range(self.MAX_INPUT_SLOTS):
            img = self.slots.get(i)
            if img and img.is_loaded:
                img.apply_resize(target, policy.preserve_aspect)
                affected.append(i)

        return affected


class ImageStore:
    """
    Global store for all user sessions.

    Thread-safe access is deferred to Phase 4 (JobManager).
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create a new session and return it."""
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        return self._sessions.pop(session_id, None) is not None


# Global singleton
image_store = ImageStore()
