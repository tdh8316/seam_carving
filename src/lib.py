import time
from copy import deepcopy

import numpy as np
import cv2
from typing import Optional, Callable
from tqdm import trange

from src.algorithms.carving import carve_seam, carve_seam_enlarge
from src.algorithms.energy import EnergyCalculator
from src.algorithms.seam import SeamFinder, draw_seam


class Image(object):
    @classmethod
    def from_path(cls, path: str):
        try:
            mat = cv2.imread(path)
        except Exception as e:
            raise ValueError(f"Failed to read '{path}': {e}")

        return cls(mat)

    def __init__(self, mat: np.ndarray):
        self._mat = deepcopy(mat)
        self._validate_mat()

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray):
        self._mat = value
        self._validate_mat()

    @property
    def shape(self) -> tuple:
        return self.mat.shape

    def _validate_mat(self):
        if not isinstance(self.mat, np.ndarray):
            raise ValueError(
                f"Expected: `np.ndarray` for `mat`, but got: {type(self.mat)}"
            )

        if len(self.mat.shape) != 3:
            raise ValueError(
                f"Image must be of shape (H, W, 3), but got: {self.mat.shape}"
            )

        if self.mat.dtype != np.uint8:
            raise ValueError(f"Image must be of type uint8, but got: {self.mat.dtype}")

    def save(self, path: str):
        cv2.imwrite(path, self.mat)

    def show(self, title: str = "Image", wait: bool = False):
        cv2.imshow(title, self.mat)
        if wait:
            cv2.waitKey(0)


class CarvableImage(object):
    """
    Class for an image to perform seam-based operations.
    """

    @classmethod
    def from_path(cls, path: str):
        return cls(Image.from_path(path))

    def __init__(
        self,
        img: Image,
        energy_function: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = EnergyCalculator.squared_diff,
        seam_function: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = SeamFinder.find_seam,
    ):
        self._img = img

        self._energy_function = energy_function
        self._seam_function = seam_function

        self._validate_functions()

    def _validate_functions(self):
        # TODO: Validate the energy and seam functions
        ...

    @property
    def img(self) -> Image:
        return self._img

    @property
    def energy_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._energy_function

    @energy_function.setter
    def energy_function(self, value: Callable[[np.ndarray], np.ndarray]):
        self._energy_function = value
        self._validate_functions()

    @property
    def seam_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._seam_function

    @seam_function.setter
    def seam_function(self, value: Callable[[np.ndarray], np.ndarray]):
        self._seam_function = value
        self._validate_functions()

    def seam_carve(
        self,
        num_seams: int,
        show_progress: bool = False,
    ) -> "CarvableImage":
        carved: np.ndarray = self.img.mat.copy()

        it = trange(num_seams, ncols=100) if show_progress else range(num_seams)

        for _ in it:
            energy_map = self.energy_function(carved)
            seam = self.seam_function(energy_map)
            carved = carve_seam(carved, seam)

        return CarvableImage(
            Image(carved),
            self.energy_function,
            self.seam_function,
        )
        
        
    def _detect_faces(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        return faces


    def _protect_faces_in_energy_map(self, energy, faces):
        for x, y, w, h in faces:
            energy[y : y + h, x : x + w] = np.max(energy) * 10
        return energy

        
    def seam_carve_with_mask(
        self,
        num_seams: int,
        show_progress: bool = False,
    ) -> "CarvableImage":
        carved: np.ndarray = self.img.mat.copy()

        it = trange(num_seams, ncols=100) if show_progress else range(num_seams)

        for _ in it:
            mask = self._detect_faces(carved)
            energy_map = self.energy_function(carved)
            energy_map = self._protect_faces_in_energy_map(energy_map, mask)
            seam = self.seam_function(energy_map)
            carved = carve_seam(carved, seam)

        return CarvableImage(
            Image(carved),
            self.energy_function,
            self.seam_function,
        )
        
    

    def seam_carve_enlarge(
        self,
        num_seams: int,
        show_progress: bool = False,
    ) -> "CarvableImage":
        enlarged: np.ndarray = self.img.mat.copy()
        carved: np.ndarray = self.img.mat.copy()

        it = trange(num_seams, ncols=100) if show_progress else range(num_seams)

        seams_to_insert = []
        for _ in it:
            energy_map = self.energy_function(carved)
            seam = self.seam_function(energy_map)
            carved = carve_seam(carved, seam)

            adjusted_seams = []
            for s in seams_to_insert:
                adjusted_seam = s.copy()
                for i in range(len(seam)):
                    if adjusted_seam[i] + 1 >= seam[i]:
                        adjusted_seam[i] += 1
                adjusted_seams.append(adjusted_seam)

            adjusted_seams.append(seam)
            seams_to_insert = adjusted_seams.copy()

        for seam in reversed(seams_to_insert):
            enlarged = carve_seam_enlarge(enlarged, seam)

        return CarvableImage(
            Image(enlarged),
            self.energy_function,
            self.seam_function,
        )

    def interactive_seam_carve(
        self,
        num_seams: int,
        title: str = "Interactive Seam Carving",
    ) -> "CarvableImage":
        carved: np.ndarray = self.img.mat.copy()

        for _ in range(num_seams):
            energy_map = self.energy_function(carved)
            seam = self.seam_function(energy_map)
            seam_img = draw_seam(carved, seam)
            cv2.imshow(title, seam_img)
            cv2.waitKey(10)
            carved = carve_seam(carved, seam)

        cv2.destroyWindow(title)

        return CarvableImage(
            Image(carved),
            self.energy_function,
            self.seam_function,
        )
