import moderngl
import numpy as np
from PIL import Image
from typing import Tuple

# TODO: for now lets be a bit dirty and just save the sandbox functions


class Render:
    """
    A dirty wrapper for some openGL function.
    """

    def __init__(self,
                 type_projection: str = "orthographic",
                 **kwargs):
        """.
        Define the properties of the render, to be reuse multiple time with
        the same configuration.


        Args:
            type_proj (str): Type of projection to render.
                Must be either "orthographic" or "perspective".
            **kwargs (dict): Dictionary of render parameters.
                rotation (array-like): Rotation matrix of the surface.

                fovy (float): Field of View Y, in radians.
                    Controls the vertical viewing angle.
                    Smaller values zoom in; larger values zoom out
                    (wide-angle).
                    Typically between 30° and 60°.
                aspect (float): Aspect ratio of the render.
                near (float): Near clipping plane.
                    The closest distance along the camera's view where objects
                    are still rendered.
                    Must be greater than 0, commonly around 0.1.
                far (float): Far clipping plane.
                    The farthest distance where objects are still visible.
        """
        self.rotation = kwargs.get("rotation", (0.0, 0.0, 0.0))
        if type_projection not in ["orthographic", "perspective"]:
            raise ValueError(f"Projection type {type_projection}"
                             "is not accepted.")
        self.type_proj = type_projection
        self.fovy = kwargs.get("fovy", np.radians(10.0))
        self.aspect = kwargs.get("aspect", 1)
        self.near = kwargs.get("near", 0.1)
        self.far = kwargs.get("far", 100.0)

    def render_textured_surface(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Render a textured surface using the given meshgrid coordinates and
        image.

        The surface is defined by three normalized meshgrids (`xx`, `yy`, `zz`)
        in the range [0, 1] and is textured with the given RGB image. The
        rendering uses the current camera and projection settings defined in
        the object (`rotation`, `type_proj`, `fovy`, `aspect`).

        Args:
            xx (np.ndarray): Normalized meshgrid of x-coordinates of the
                surface.
            yy (np.ndarray): Normalized meshgrid of y-coordinates of
                the surface.
            zz (np.ndarray): Normalized meshgrid of z-coordinates
                of the surface.
            image (np.ndarray): RGB image to be mapped onto the
                surface. Must have the same height and width as
                the meshgrids.

        Returns:
            np.ndarray: Rendered RGB image as a NumPy array (dtype `uint8`).

        Raises:
            AssertionError: If the image dimensions do not match the meshgrids.
        """
        rotation = self.rotation
        type_proj = self.type_proj
        fovy = self.fovy
        aspect = self.aspect

        H, W = xx.shape
        assert image.shape[:2] == (
            H, W), "Image must match meshgrid dimensions"

        # Scale meshgrid from [0, 1] to OpenGL [-1, 1]
        xx = np.copy(xx) * 2 - 1
        yy = np.copy(yy) * 2 - 1
        zz = np.copy(zz) * 2 - 1

        # Create OpenGL context and framebuffer
        ctx = moderngl.create_standalone_context()
        fbo = ctx.simple_framebuffer((W, H))
        fbo.use()

        # Prepare texture
        tex_img = Image.fromarray(image).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM)
        texture = ctx.texture((W, H), 3, tex_img.tobytes())
        texture.build_mipmaps()
        texture.use(location=0)

        # Build vertex buffer (x, y, z, u, v)
        vertices = []
        for i in range(H):
            for j in range(W):
                x, y, z = xx[i, j], yy[i, j], zz[i, j]
                u = j / (W - 1)
                v = i / (H - 1)
                vertices.extend([x, y, z, u, v])
        vertices = np.array(vertices, dtype="f4")

        # Build index buffer
        indices = []
        for i in range(H - 1):
            for j in range(W - 1):
                idx = i * W + j
                indices.extend(
                    [idx, idx + 1, idx + W, idx + 1, idx + W + 1, idx + W])
        indices = np.array(indices, dtype="i4")

        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(indices.tobytes())

        # Compile shaders
        prog = ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_position;
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(tex, v_texcoord);
                }
            """,
        )

        vao = ctx.vertex_array(
            prog, [(vbo, "3f 2f", "in_position", "in_texcoord")], ibo
        )

        # Compute projection-view matrix
        xmin, xmax = xx.min(), xx.max()
        ymin, ymax = yy.min(), yy.max()
        zmin, zmax = zz.min(), zz.max()

        rot = self.rotation_matrix(*rotation)

        if type_proj == "orthographic":
            proj = self.orthographic(
                xmin, xmax, ymin, ymax, zmin - 1, zmax + 1)
            mvp = proj @ rot
        else:
            view, _, dist = self.compute_camera_view(xx, yy, zz, fovy, aspect)
            proj = self.perspective(fovy, aspect, 0.1, dist * 2.0)
            mvp = proj @ view @ rot

        prog["mvp"].write(mvp.T.tobytes())  # type: ignore

        # Render
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        # Read back image
        data = fbo.read(components=3, alignment=1)
        img = Image.frombytes("RGB", (W, H), data)
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        return np.array(img)

    def orthographic(
        self,
        left: float,
        right: float,
        bottom: float,
        top: float,
        near: float,
        far: float,
    ) -> np.ndarray:
        """
        Compute the orthographic projection matrix.

        This projection maps coordinates directly to the screen without
        perspective distortion, keeping parallel lines parallel.
        It is defined by the bounds of the viewing volume
        (`left`, `right`, `bottom`, `top`, `near`, `far`).

        Args:
            left (float): Position of the left clipping plane.
            right (float): Position of the right clipping plane.
            bottom (float): Position of the bottom clipping plane.
            top (float): Position of the top clipping plane.
            near (float): Near clipping plane. Must be > 0.
            far (float): Far clipping plane.

        Returns:
            np.ndarray: A 4×4 orthographic projection matrix (dtype `float32`).
        """
        return np.array(
            [
                [2 / (right - left), 0, 0, -(right + left) / (right - left)],
                [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                [0, 0, 0, 1],
            ],
            dtype="f4",
        )

    def perspective(
        self,
        fovy: float,
        aspect: float,
        near: float,
        far: float
    ) -> np.ndarray:
        """
        Compute the perspective projection matrix.

        This projection simulates a camera with perspective, making objects
        appear smaller as they get farther away. It is defined by the vertical
        field of view (`fovy`), aspect ratio, and near/far clipping planes.

        Args:
            fovy (float): Field of view in the Y direction, in radians.
                Smaller values zoom in; larger values zoom out (wide-angle).
                Typically between 30° and 60°.
            aspect (float): Aspect ratio of the render (width / height). near
            (float): Near clipping plane. Must be > 0. far (float): Far
            clipping plane.

        Returns:
            np.ndarray: A 4×4 perspective projection matrix (dtype `float32`).
        """
        f = 1.0 / np.tan(fovy / 2.0)
        return np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far),
                 (2 * far * near) / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype="f4",
        )

    def rotation_matrix(
        self,
        yaw: float,
        pitch: float,
        roll: float
    ) -> np.ndarray:
        """
        Compute a 3D rotation matrix from yaw, pitch, and roll angles.

        Rotations are applied in the following order:
        1. Pitch (rotation around the x-axis)
        2. Yaw (rotation around the y-axis)
        3. Roll (rotation around the z-axis)

        Args:
            yaw (float): Rotation angle around the y-axis, in radians.
            pitch (float): Rotation angle around the x-axis, in radians.
            roll (float): Rotation angle around the z-axis, in radians.

        Returns:
            np.ndarray: A 4×4 rotation matrix (dtype `float32`).
        """
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1],
            ]
        )

        # Apply rotations: pitch → yaw → roll
        R = Rz @ Ry @ Rx

        R4 = np.eye(4, dtype="f4")
        R4[:3, :3] = R
        return R4

    def compute_camera_view(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray,
        fovy: float,
        aspect_ratio: float,
        margin: float = 1.1,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute a camera view matrix that fits the 3D object defined by the
        meshgrid coordinates.

        The function determines a camera position and orientation such that
        the entire object is visible within the given vertical field of view
        and aspect ratio, with an optional margin.

        Args:
            xx (np.ndarray): Meshgrid of x-coordinates of the surface.
            yy (np.ndarray): Meshgrid of y-coordinates of the surface.
            zz (np.ndarray): Meshgrid of z-coordinates of the surface.
            fovy (float): Vertical field of view in radians.
            aspect_ratio (float): Aspect ratio of the render (width / height).
            margin (float, optional): Factor controlling extra space around
                the object.
                A value of `1.1` means 10% margin. Defaults to `1.1`.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - view_matrix: A 4×4 camera view matrix (dtype `float32`).
                - center: The 3D coordinates of the object center.
                - distance: Distance from the camera to the object.
        """
        # 1. Compute bounding box
        xmin, xmax = np.min(xx), np.max(xx)
        ymin, ymax = np.min(yy), np.max(yy)
        zmin, zmax = np.min(zz), np.max(zz)

        # 2. Object center
        center = np.array(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])

        # 3. Object dimensions
        width = xmax - xmin
        height = ymax - ymin

        # 4. Compute required camera distance
        obj_height = max(height, width / aspect_ratio) * margin
        distance = obj_height / (2 * np.tan(fovy / 2))

        # 5. View matrix: translate to center, then move back along -Z
        view = np.eye(4, dtype=np.float32)
        view[:3, 3] = -center
        view = view @ self.translation_matrix(0, 0, -distance)

        return view, center, distance

    def translation_matrix(self,
                           x: float,
                           y: float,
                           z: float) -> np.ndarray:
        """
        Create a 4×4 translation matrix.

        This matrix translates points or objects in 3D space by the
        given offsets along the x, y, and z axes.

        Args:
            x (float): Translation offset along the x-axis.
            y (float): Translation offset along the y-axis.
            z (float): Translation offset along the z-axis.

        Returns:
            np.ndarray: A 4×4 translation matrix (dtype `float32`).
        """
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z
        return mat
