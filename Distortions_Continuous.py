import numpy as np
import random
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates, zoom
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2

# ------------------ CORE IDEA ------------------
# Every parameter is scaled directly by a ∈ [0,1]

# ---------- CT ----------

def ct_kernel(img, a):
    img = (img - img.min()) / (img.max() + 1e-8)
    H,W = img.shape

    k = fftshift(fft2(img))
    y,x = np.ogrid[:H,:W]
    r = np.sqrt((x-W//2)**2 + (y-H//2)**2)
    r_norm = r / r.max()

    sigma = 0.8 - 0.7*a
    mask = np.exp(-(r_norm / (sigma+1e-6))**2)

    k_mod = k * mask
    out = np.real(ifft2(ifftshift(k_mod)))

    noise = gaussian_filter(np.random.normal(0, 0.05*a, img.shape), 10*a+1)
    return np.clip(out + noise, 0, 1).astype(np.float32)


# ---------- MRI ----------

def motion(img, a):
    if a == 0: return img.copy()

    H,W = img.shape
    num = int(1 + 6*a)
    angle = 30*a
    shift = 10*a

    K = []
    for _ in range(num):
        ang = np.deg2rad(random.uniform(-angle, angle))
        tx = random.uniform(-shift, shift)
        ty = random.uniform(-shift, shift)

        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang),  np.cos(ang)]])
        A = np.linalg.inv(R)

        center = np.array([H/2,W/2])
        offset = center - A.dot(center) - np.array([ty,tx])

        warped = affine_transform(img, A, offset=offset, order=3)
        K.append(fftshift(fft2(warped)))

    out = np.zeros_like(K[0])
    lines = H // num
    for i in range(H):
        out[i] = K[min(i//lines, num-1)][i]

    return np.abs(ifft2(ifftshift(out))).astype(np.float32)


def mri_noise(img, a):
    img = img / (img.max()+1e-8)
    poisson_scale = 1 + 10*(1-a)
    gauss = np.random.normal(0, 0.2*a, img.shape)
    return np.clip(img + gauss, 0, 1)


def b0(img, a):
    H,W = img.shape
    field = gaussian_filter(np.random.randn(H,W), 30*(1-a)+5)
    field = field / (np.abs(field).max()+1e-8)

    phase = 4*np.pi*a*field
    k = fft2(img) * np.exp(1j*phase)
    return np.clip(np.abs(ifft2(k)),0,1)


# ---------- SHARED ----------

def elastic(img, a):
    alpha = 10*a
    sigma = 20*(1-a)+5

    dx = gaussian_filter((np.random.rand(*img.shape)*2-1), sigma)*alpha
    dy = gaussian_filter((np.random.rand(*img.shape)*2-1), sigma)*alpha

    x,y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    coords = np.array([y+dy, x+dx])
    return map_coordinates(img, coords, order=1)


def perspective(img, a):
    H,W = img.shape
    offset = int(25*a)

    src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    dst = src + np.random.randint(-offset,offset+1,(4,2)).astype(np.float32)

    M = cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img,M,(W,H))


def intensity(img, a):
    factor = 1 - 0.7*a
    return np.clip(img*factor,0,1)


def shift(img, a):
    tx = 10*a
    ty = 10*a
    return affine_transform(img, np.eye(2), offset=[-ty,-tx])


def rotate(img, a):
    angle = 30*a
    H,W = img.shape

    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta),-np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    A = np.linalg.inv(R)

    center = np.array([H/2,W/2])
    offset = center - A.dot(center)

    return affine_transform(img, A, offset=offset)


# ---------- DICTS ----------

CT_DIST = {
    "kernel": ct_kernel,
    "elastic": elastic,
    "perspective": perspective,
    "intensity": intensity,
    "shift": shift,
    "rotate": rotate
}

MRI_DIST = {
    "motion": motion,
    "noise": mri_noise,
    "b0": b0,
    "elastic": elastic,
    "perspective": perspective,
    "intensity": intensity,
    "shift": shift,
    "rotate": rotate
}