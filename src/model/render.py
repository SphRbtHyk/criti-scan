import moderngl
import numpy as np
from PIL import Image
from typing import Tuple

# for now lets be a bit dirty and just save the sandbox functions


class Render:
    """
    A dirty wrapper of some openGL function
    """

    def __init__(self, **kwargs):
        """
        Define the properties of the render, to be reuse multiple time with the same configuration

        kwargs: dict
            A dictionnary of render parameters
            rotation – The rotation matrix of the surface
            type_proj - "orthographic" or "perspective"
                define the type of projection to render
            fovy – Field of View Y (in radians)
                It controls how wide the camera sees vertically.
                A smaller fovy zooms in, while a larger one zooms out (like a wide-angle lens).
                Typically set between 30° and 60°
            aspect - The aspect ratio of the render
            near – Near Clipping Plane
                The closest distance (along the camera's view) where objects are still rendered.
                Anything closer than this will be clipped and not appear.
                Must be > 0, commonly around 0.1.
            far – Far Clipping Plane
                The farthest distance where objects are still visible.
                Anything beyond this will not be rendered.
        """
        self.rotation = kwargs.get("rotation", (0.0, 0.0, 0.0))
        self.type_proj = kwargs.get("type_proj", "orthographic")
        self.fovy = kwargs.get("fovy", np.radians(10.0))
        self.aspect = kwargs.get("aspect", 1)
        self.near = kwargs.get("near", 0.1)
        self.far = kwargs.get("far", 100.0)

    def render_textured_surface(
        self, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, image: np.ndarray
    ) -> np.ndarray:
        """
        Render the scene made of a given image on a given surface defined by its meshgrid, with the defined properties of the render

        parameters:
        ===========
        xx: meshgrid
            The coordinates of the surface in x, normalized.
        yy: meshgrid
            The coordinates of the surface in y, normalized.
        zz: meshgrid
            The coordinates of the surface in z, normalized.
        image: ndarray
            The image to be mapped on the surface

        Return:
        =======
        ndarray:
            The render as an ndarray

        """

        rotation = self.rotation
        type_proj = self.type_proj
        fovy = self.fovy
        aspect = self.aspect
        near = self.near
        far = self.far

        H, W = xx.shape
        assert image.shape[:2] == (H, W), "Image must match meshgrid dimensions"

        # scale the meshgrid
        # as we work here from -1 to 1 and the given meshgrid are normalized
        xx = np.copy(xx) * 2 - 1
        yy = np.copy(yy) * 2 - 1
        zz = np.copy(zz) * 2 - 1

        # Create OpenGL context and framebuffer
        ctx = moderngl.create_standalone_context()
        fbo = ctx.simple_framebuffer((W, H))
        fbo.use()

        # Prepare texture
        tex_img = Image.fromarray(image).transpose(Image.FLIP_TOP_BOTTOM)
        texture = ctx.texture((W, H), 3, tex_img.tobytes())
        texture.build_mipmaps()
        texture.use(location=0)

        # Create vertex buffer
        vertices = []
        for i in range(H):
            for j in range(W):
                x, y, z = xx[i, j], yy[i, j], zz[i, j]
                u = j / (W - 1)
                v = i / (H - 1)
                vertices.extend([x, y, z, u, v])
        vertices = np.array(vertices, dtype="f4")

        indices = []
        for i in range(H - 1):
            for j in range(W - 1):
                idx = i * W + j
                indices.extend([idx, idx + 1, idx + W, idx + 1, idx + W + 1, idx + W])
        indices = np.array(indices, dtype="i4")

        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(indices.tobytes())

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

        # Bounding box of surface
        xmin, xmax = xx.min(), xx.max()
        ymin, ymax = yy.min(), yy.max()
        zmin, zmax = zz.min(), zz.max()

        rot = self.rotation_matrix(*rotation)

        if type_proj == "orthographic":
            proj = self.orthographic(xmin, xmax, ymin, ymax, zmin - 1, zmax + 1)
            mvp = proj @ rot

        else:
            view, center, dist = self.compute_camera_view(xx, yy, zz, fovy, aspect)
            proj = self.perspective(fovy, aspect, 0.1, dist * 2.0)
            mvp = proj @ view @ rot

        prog["mvp"].write(mvp.T.tobytes())

        # Render
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        # Read back image
        data = fbo.read(components=3, alignment=1)
        img = Image.frombytes("RGB", (W, H), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

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
        Define the parameters for an orthographic render

        Parameters
        ==========
        left: float
            The position of the left corner
        right: float
            The position of the right corner
        bottom: float
            The position of the bottom corner
        top: float
            The position of the top corner
        aspect: float
            The aspect ratio of the render
        near: float
            Near Clipping Plane
            The closest distance (along the camera's view) where objects are still rendered.
            Anything closer than this will be clipped and not appear.
            Must be > 0, commonly around 0.1.
        far: float
            Far Clipping Plane
            The farthest distance where objects are still visible.
            Anything beyond this will not be rendered.

        Return
        ======
            np.array
                the parameters as an array

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
        self, fovy: float, aspect: float, near: float, far: float
    ) -> np.array:
        """
        Define the parameters for an perspective render

        Parameters
        ==========
        fovy – Field of View Y (in radians)
            It controls how wide the camera sees vertically.
            A smaller fovy zooms in, while a larger one zooms out (like a wide-angle lens).
            Typically set between 30° and 60°
        aspect - The aspect ratio of the render
        near – Near Clipping Plane
            The closest distance (along the camera's view) where objects are still rendered.
            Anything closer than this will be clipped and not appear.
            Must be > 0, commonly around 0.1.
        far – Far Clipping Plane
            The farthest distance where objects are still visible.
            Anything beyond this will not be rendered.

        Return
        ======
            np.array
                the parameters as an array
        """
        f = 1.0 / np.tan(fovy / 2.0)
        return np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype="f4",
        )

    def rotation_matrix(self, yaw: float, pitch: float, roll: float) -> np.array:
        """
        Define a 3D rotation matrix, with its yaw(y), pitch(x) and roll(z)
        The matrix follow this succession of rotation: along x, y then z

        Parameters
        ==========
        yaw: float
            The rotation along y
        pitch: float
            The rotation along x
        roll: float
            The rotation along z
        Return
        ======
            R4: np.array
                the total rotation matrix
        """
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        Ry = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
        )
        Rz = np.array(
            [
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1],
            ]
        )
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
        Compute a camera view matrix to fit the 3D object defined by xx, yy, zz

        Parameters:
        ===========
            xx, yy, zz: np.ndarray
                Meshgrid arrays of the surface
            fovy: float
                Vertical field of view in radians
            aspect_ratio: float
                width / height of render
            margin: float
                How much extra space around the object (1.1 = 10% margin)

        Returns:
        ========
            view_matrix: np.ndarray
                4x4 camera view matrix
            center: np.ndarray
                Center of the object (for lookat)
            distance: float
                Distance from camera to object
        """

        # 1. Compute bounding box
        xmin, xmax = np.min(xx), np.max(xx)
        ymin, ymax = np.min(yy), np.max(yy)
        zmin, zmax = np.min(zz), np.max(zz)

        # 2. Object center
        center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])

        # 3. Object size
        width = xmax - xmin
        height = ymax - ymin

        # 4. Convert fovy to radians
        # already done

        # 5. Compute required camera distance
        obj_height = max(height, width / aspect_ratio) * margin
        distance = obj_height / (2 * np.tan(fovy / 2))

        # 6. View matrix: move camera along -Z, looking at object center
        view = np.eye(4, dtype=np.float32)
        view[:3, 3] = -center
        view = view @ self.translation_matrix(0, 0, -distance)

        return view, center, distance

    def translation_matrix(self, x: float, y: float, z: float) -> np.array:
        """
        Creates a 4x4 translation matrix for the camera space

        Parameters
        ==========
        x: float
            The translation along x
        y: float
            The translation along y
        z: float
            The translation along z
        Return
        ======
            mat: np.array
                the total translation matrix
        """
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z

        return mat
