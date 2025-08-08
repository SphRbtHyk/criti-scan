import numpy as np
from PIL import Image
from typing import Tuple
import sympy
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad


"""
Inspired form the blog page https://mzucker.github.io/2016/08/15/page-dewarping.html
"""
class BookDeform:
    """
    This class generate a surface deformed by a BSpline and compute the coordinates transfer for a texture to be wrapped on it
    """

    def __init__(self, width, height, alpha, beta):
        self.xx, self.yy, self.zz = self.generate_flat_page(width=width, height=height)
        self.xx_deformed, self.yy_deformed, self.zz_deformed, self.xx_scaled = self.deform_page(self.xx, self.yy, alpha, beta)

    def get_surface(self):
        """
        get the normalized meshgrid of the surface.
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Meshgrids (X, Y, Z) representing a flat page in 3D space, 
                with Z set to the deformation field. All normalized.
        """
        # the deform is proportionnal to the width, i.e. the "x" component
        return self.xx_deformed / np.max(self.xx_deformed),\
        self.yy_deformed / np.max(self.yy_deformed),\
        self.zz_deformed / np.max(self.xx_deformed)
    
    def generate_flat_page(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a flat 2D rectangular page as a meshgrid.
        
        Args:
            width (int): Width of the page (in pixel).
            height (int): Height of the page (in pixel).
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Meshgrids (X, Y, Z) representing a flat page in 3D space, 
                with Z set to zero everywhere.
        """
        x = np.linspace(0, width, width, endpoint = False)
        y = np.linspace(0, height, height, endpoint = False)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)  # Flat surface (no deformation)
        return xx, yy, zz
    
    
    def generate_bspline(self, num_alpha: float, num_beta: float, xvals: np.ndarray) -> np.ndarray:
        """
        Generate a custom cubic B-spline curve constrained by derivative values at its endpoints.
        
        The spline is defined as a cubic polynomial f(x) = ax^3 + bx^2 + cx + d, where:
          - f(0) = 0 and f(1) = 0 (start and end at zero)
          - f'(0) = alpha (controls initial slope)
          - f'(1) = beta  (controls final slope)
        
        These constraints create a flexible spline that can simulate local bending.
    
        Args:
            num_alpha (float): Desired slope (derivative) at x = 0.
            num_beta (float): Desired slope at x = 1.
            xvals (np.ndarray): 1D array of x-values at which to evaluate the spline.
    
        Returns:
            deform_vals : np.ndarray: 1D array of y-values corresponding to the spline evaluated at xvals.
            integral : np.ndarray: 1D array of the positive integral of the spline along the way
        """
        
        # Define symbolic variables
        a, b, c, d, x, alpha, beta = sympy.symbols('a b c d x alpha beta')
        
        # Define the general cubic polynomial: f(x) = ax^3 + bx^2 + cx + d
        f = a * x**3 + b * x**2 + c * x + d
        
        # Compute the first derivative: f'(x)
        fp = f.diff(x)
        
        # Apply boundary conditions:
        #   f(0) = 0   → ensures the curve starts at y = 0
        #   f(1) = 0   → ensures the curve ends at y = 0
        #   f'(0) = alpha  → controls starting slope
        #   f'(1) = beta   → controls ending slope
        conditions = [
            f.subs(x, 0),                # f(0) = 0
            f.subs(x, 1),                # f(1) = 0
            fp.subs(x, 0) - alpha,       # f'(0) = alpha
            fp.subs(x, 1) - beta         # f'(1) = beta
        ]
        
        # Solve for coefficients a, b, c, d in terms of alpha and beta
        solution = sympy.solve(conditions, [a, b, c, d])
    
        # Substitute numeric values of alpha and beta
        subs_dict = {alpha: num_alpha, beta: num_beta}
        coeffs = [solution[coef].subs(subs_dict) for coef in [a, b, c, d]]
    
        # Convert symbolic coefficients to numeric
        coeffs = [float(c) for c in coeffs]
    
        # Evaluate the resulting polynomial at xvals
        deform_vals = np.polyval(coeffs, xvals/np.max(xvals)) * np.max(xvals)
    
        nb_point = xvals.shape[1]
        integral = np.zeros(nb_point)
        for x in range(nb_point):
            if x == 0:
                integral[0] = 0
            else:
                integral[x] = integral[x-1] + self.curve_length_cubic(coeffs, (x-1)/np.max(xvals), x/np.max(xvals)) * np.max(xvals)
        
        return deform_vals, integral
    
    def curve_length_cubic(self, coeffs: np.ndarray, a: float, b: float) -> float:
        """
        Compute the arc length of a cubic polynomial curve y = f(x) between x=a and x=b.
    
        Args:
            coeffs (np.ndarray): Coefficients [d, c, b, a] for the cubic polynomial ax^3 + bx^2 + cx + d.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.
    
        Returns:
            float: Arc length of the curve from x=a to x=b.
        """
        # Construct the polynomial: p(x) = a*x^3 + b*x^2 + c*x + d
        p = Polynomial(coeffs)
    
        # Compute its derivative: p'(x)
        dp = p.deriv()
    
        # Define the integrand: sqrt(1 + (p'(x))^2)
        def integrand(x):
            return np.sqrt(1 + dp(x)**2)
    
        # Numerically integrate the arc length
        length, _ = quad(integrand, a, b)
    
        return length


    def deform_page(self, xx: np.ndarray, yy: np.ndarray, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a 2-parameters constraint spline to create a 3D warp (to a latter apply to a flat page to simulate bending).
    
        Args:
            xx (np.ndarray): X-coordinates of the meshgrid.
            yy (np.ndarray): Y-coordinates of the meshgrid.
            alpha (float): Desired slope (derivative) at x = 0.
            beta (float): Desired slope at x = 1.
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Deformed 3D surface coordinates (X, Y, Z).
        """
        zz, integral = self.generate_bspline(alpha, beta, xx)  # Spline wave deformation in Z direction
        # now we deform xx to fit the same surface of the page
        xx_scaled =  np.copy(xx)
        xx_scaled, _ = np.meshgrid(integral, yy[:, 0])
    
        return xx, yy, zz, xx_scaled
    
    def project_image_on_surface(self, image_to_distord):
        """
        Apply a transform to map the given image to the deformed surface.
    
        Args:
            image_to_distord (np.ndarray): The image to distord.

        Returns:
            np.ndarray: Deformed image to be mapped on the 3D surface coordinates (X, Y, Z).
        """
        width = image_to_distord.shape[1]
        height = image_to_distord.shape[0]
        # it will as well copy the color channel if present
        img_proj = np.zeros(image_to_distord.shape, dtype=np.uint8)

        xx_ref = self.xx_scaled.astype(int) 
        for yy in range(height):
            for xx in range(width):
                new_x = np.where(xx_ref[yy, :] == xx)[-1]
                new_y = yy
                img_proj[new_y, new_x] = image_to_distord[yy, xx]

        return img_proj
