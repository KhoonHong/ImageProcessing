from django.shortcuts import render, redirect, get_object_or_404
from .models import UploadedImage
from django.http import JsonResponse
from django.http import HttpResponse
import cv2
import os
from django.views.generic.edit import FormView
from .forms import FileFieldForm
import numpy as np
import uuid
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import sweetify

original_image_path_random, original_image_path_reference = None, None
SIFT_sample_keypoints, SIFT_sample_descriptors = None, None

@method_decorator(csrf_exempt, name='dispatch')
class UpdateLabelView(View):
    def post(self, request, *args, **kwargs):
        try:
            image_ids = request.POST.get('image_ids').split(',')
            print(image_ids)
            label = request.POST.get('label')
            images = UploadedImage.objects.filter(id__in=image_ids)
            for image in images:
                image.image_label = label
                image.save()
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

class FileFieldFormView(FormView):
    form_class = FileFieldForm
    template_name = "upload.html"  # Replace with your template.

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

    def form_valid(self, form):
        files = form.cleaned_data["file_field"]
        for f in files:
            image_instance = UploadedImage(image=f)
            image_instance.save()
        return super().form_valid()

def upload_view(request):
    """Handle the multi-image upload view."""
    if request.method == 'POST':
        form = FileFieldForm(request.POST, request.FILES)
        if form.is_valid():
            files = form.cleaned_data["file_field"]
            for f in files:
                image_instance = UploadedImage(image=f)
                image_instance.save()
            return redirect('upload_view')
    else:
        form = FileFieldForm()

    images = UploadedImage.objects.all()

    return render(request, 'upload.html', {'form': form, 'images': images})


def delete_images(request):
    """Handle deletion of multiple images."""
    if request.method == 'POST':
        image_ids_string = request.POST.getlist('image_ids')[0]
        image_ids = [int(x) for x in image_ids_string.split(',')]

        if not image_ids:
            return JsonResponse({'success': False, 'message': 'No image IDs provided!'})

        # delete according to the image IDs
        count_deleted, _ = UploadedImage.objects.filter(id__in=image_ids).delete()
        if count_deleted:
            sweetify.success(request, f'{count_deleted} image(s) deleted successfully!', button='Ok', timer=3000)
            return JsonResponse({'success': True, 'message': f'{count_deleted} image(s) deleted successfully!'})
        else:
            return JsonResponse({'success': False, 'message': 'No images found for the provided IDs!'})

def morphological_methods(request):
    # randomly grab one image
    image = UploadedImage.objects.filter(image_label='NULL').order_by('?').first()
    print(image.image)
    return render(request, 'morphological_methods.html', {'image': image})

def morph_demo(request):
    if request.method != "POST":
        return JsonResponse({'success': False, 'message': 'Invalid method'})

    form_data = request.POST
    image_url = form_data.get('image').split('/')[-1]
    image_url = os.path.join('media/uploaded_images', image_url)

    if not os.path.exists(image_url):
        return JsonResponse({'success': False, 'message': 'Image not found'})

    # Handle image loading from URL if needed
    # TODO: You might need to download the image to your local storage if you're dealing with URLs

    # For now, assuming the image_url is a path to the local file
    image_path = image_url

    # Convert to grayscale if switch is on
    if form_data.get('grayscale_switch') == 'on':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        return JsonResponse({'success': False, 'message': 'Unable to load image'})

    # Apply filtering if selected
    filter_type = form_data.get('filtering-radio', 'none')
    image = apply_filter(image_path, filter_type)

    # Apply morphological operations
    if form_data.get('erosion_switch') == 'on':
        image = apply_erosion(image)

    if form_data.get('dilation_switch') == 'on':
        image = apply_dilation(image)

    if form_data.get('opening_switch') == 'on':
        image = morphological_operations(image_path, 'opening')

    if form_data.get('closing_switch') == 'on':
        image = morphological_operations(image_path, 'closing')

    # TODO: Handle saving or displaying the processed image, if needed.

    return JsonResponse({
        'success': True, 
        'message': 'Image processed successfully!', 
        'image_url': os.path.split(image_url)[-1],
    })


def apply_filter(image_path, filter_type):
    """
    Apply specified filter to the input image.
    
    Parameters:
    - image_path: Path to the image file.
    - filter_type: Type of filter to apply ('gaussian', 'median', or 'none').
    
    Returns:
    - filtered_image: Image after applying the specified filter.
    """
    
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")

    # Apply the specified filter
    if filter_type == 'Gaussian Filters':
        # Gaussian blur
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'Median Filters':
        # Median blur
        filtered_image = cv2.medianBlur(image, 5)
    elif filter_type == 'No Filters':
        # No filter, return original image
        filtered_image = image
    else:
        raise ValueError(f"Invalid filter type: {filter_type}. Choose from 'gaussian', 'median', or 'none'.")
    
    return filtered_image

def morphological_operations(image_path, operation, kernel_size=5):
    """
    Perform morphological operations (opening/closing) on an image.

    Parameters:
    - image_path: Path to the input image.
    - operation: String, either 'opening' or 'closing'.
    - kernel_size: Integer, the size of the structuring element. Default is 5.

    Returns:
    - Processed image.
    """

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check for valid operation
    if operation not in ['opening', 'closing']:
        raise ValueError("Invalid operation. Choose either 'opening' or 'closing'.")

    # Create a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == 'opening':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    else:  # operation == 'closing'
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return result

def apply_erosion(image, kernel_size=3, iterations=1):
    """
    Apply erosion on the input image.
    
    Parameters:
    - image: Input image (usually binary).
    - kernel_size: Size of the square kernel used for erosion (default is 3x3).
    - iterations: Number of times erosion is applied (default is 1).

    Returns:
    - Eroded image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image

def apply_dilation(image, kernel_size=3, iterations=1):
    """
    Apply dilation on the input image.
    
    Parameters:
    - image: Input image (usually binary).
    - kernel_size: Size of the square kernel used for dilation (default is 3x3).
    - iterations: Number of times dilation is applied (default is 1).

    Returns:
    - Dilated image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

def image_reg(request):
    # First, try to get a single image labeled as 'normal'
    reference_image = UploadedImage.objects.filter(image_label='normal').first()
    print(reference_image)
    # print all the labels from the database
    print(UploadedImage.objects.values_list('image_label', flat=True).distinct())

    # Then, fetch a random image labeled 'NULL' or 'NONE'
    random_image = UploadedImage.objects.filter(image_label__in=['NULL', 'NONE']).order_by('?').first()
    print(random_image)

    if not reference_image:
        return HttpResponse("No 'normal' labeled image available.")

    if not random_image:
        return HttpResponse("No 'NULL' or 'NONE' labeled image available.")

    return render(request, 'image_reg.html', {'reference_image': reference_image, 'random_image': random_image})
    
def get_feature_switches(form_data):
    return {
        'use_orb': form_data.get('ORB_switch') == 'on',
        'use_sift': form_data.get('SIFT_switch') == 'on',
        'use_surf': form_data.get('SURF_switch') == 'on',
    }

def feature_detect(request):
    if request.method != "POST":
        return JsonResponse({'success': False, 'message': 'Invalid method'})

    form_data = request.POST
    image_url_reference = form_data.get('image_reference').split('/')[-1]
    image_url_random = form_data.get('image_random').split('/')[-1]
    global original_image_path_reference, original_image_path_random
    original_image_path_reference = os.path.join('media/uploaded_images', image_url_reference) if original_image_path_reference == None else original_image_path_reference
    original_image_path_random = os.path.join('media/uploaded_images', image_url_random) if original_image_path_random == None else original_image_path_random

    if not os.path.exists(original_image_path_reference) or not os.path.exists(original_image_path_random):
        return JsonResponse({'success': False, 'message': 'Image not found'})

    switches = get_feature_switches(form_data)

    # If no feature detection switch is turned on, use the original image
    if not any(switches.values()):
        processed_image_path_reference = original_image_path_reference
        processed_image_path_random = original_image_path_random
    else:
        # Generate a unique filename using UUID
        unique_filename_random = f"sample_SIFT_{uuid.uuid4().hex}.jpg"
        unique_filename_reference = f"sample_SIFT_{uuid.uuid4().hex}.jpg"
        processed_image_path_random = os.path.join('media', 'uploaded_images', unique_filename_random)
        processed_image_path_reference = os.path.join('media', 'uploaded_images', unique_filename_reference)
        image_random = cv2.imread(original_image_path_random, cv2.IMREAD_GRAYSCALE)
        image_reference = cv2.imread(original_image_path_reference, cv2.IMREAD_GRAYSCALE)
        image_random = combined_keypoints(image_random, form_data, **switches)
        image_reference = combined_keypoints(image_reference, form_data, **switches)
        cv2.imwrite(processed_image_path_random, image_random)
        cv2.imwrite(processed_image_path_reference, image_reference)

    # return a json response
    return JsonResponse({
        'success': True, 
        'message': 'Image registered successfully!', 
        'image_url_random': os.path.split(processed_image_path_random)[-1], 
        'image_url_reference': os.path.split(processed_image_path_reference)[-1], 
        'parameters': form_data
    })


def image_reg_align(request):
    if request.method != "POST":
        return JsonResponse({'success': False, 'message': 'Invalid method'})
    form_data = request.POST
    global original_image_path_random, original_image_path_reference
    original_image_random = cv2.imread(original_image_path_random, cv2.IMREAD_GRAYSCALE)
    original_image_reference = cv2.imread(original_image_path_reference, cv2.IMREAD_GRAYSCALE)

    good_matches, keypoints1, keypoints2 = combined_matching(original_image_reference, original_image_random, form_data, image_matching_type=form_data.get('image-matching-radio'), **get_feature_switches(form_data))

    # Compute the transformation
    M, mask = compute_transformation(original_image_reference, original_image_random, keypoints1, keypoints2, good_matches, form_data)

    # Align the images
    aligned_image = align_images(original_image_reference, original_image_random, M, form_data)

    # save the aligned image
    unique_filename = f"aligned_image_{uuid.uuid4().hex}.jpg"
    processed_image_path = os.path.join('media', 'uploaded_images', unique_filename)
    cv2.imwrite(processed_image_path, aligned_image)

    apply_to_all(original_image_reference, M, form_data)

    return JsonResponse({'success': True, 'message': 'Image registered and aligned successfully!', 'align_image_url': os.path.split(processed_image_path)[-1], 'reference_image_url': os.path.split(original_image_path_reference)[-1]})

def apply_to_all(original_image_reference, M, form_data):
    # align all the none and null images
    none_images = UploadedImage.objects.filter(image_label=None)
    null_images = UploadedImage.objects.filter(image_label='NULL')

    for image in none_images:
        image_path = os.path.join('media', image.image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # check if the image is a valid image
        if image is None:
            continue
        aligned_image = align_images(original_image_reference, image, M, form_data)
        cv2.imwrite(image_path, aligned_image)

    for image in null_images:
        image_path = os.path.join('media', image.image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        aligned_image = align_images(original_image_reference, image, M, form_data)
        cv2.imwrite(image_path, aligned_image)

def combined_keypoints(image, form_data, use_orb=True, use_surf=True, use_sift=True):
    """
    Computes keypoints from ORB, SURF, SIFT or any combination and returns an image with the keypoints drawn.
    
    Parameters:
    - image: Input image.
    - use_orb: Boolean, whether to compute keypoints using ORB.
    - use_surf: Boolean, whether to compute keypoints using SURF.
    - use_sift: Boolean, whether to compute keypoints using SIFT.
    
    Returns:
    - image_with_keypoints: Image with keypoints drawn.
    """
    
    # Lists to collect keypoints from all algorithms
    keypoints = []

    # Use ORB if specified
    if use_orb:
        orb_kp, _ = compute_orb(image, n_features=form_data.get('ORB_nfeatures_range'), scaleFactor=form_data.get('ORB_scaleFactor_range'), nLevels=form_data.get('ORB_nLevels_range'), edgeThreshold=form_data.get('ORB_edgeThreshold_range'),  firstLevel=form_data.get('ORB_firstLevel_range'), WTA_K=form_data.get('ORB_WTAK_range'), patchSize=form_data.get('ORB_patchSize_range'), fastThreshold=form_data.get('ORB_fastThreshold_range'), scoreType=True if form_data.get('ORB_scoreType_radio') == 'on' else False)
        keypoints.extend(orb_kp)
    
    # Use SURF if specified
    if use_surf:
        surf_kp, _ = compute_surf(image, hessianThreshold=form_data.get('SURF_hessianThreshold_range'),  nOctaves=form_data.get('SURF_nOctaves_range'), nOctaveLayers=form_data.get('SURF_nOctaveLayers_range'),  extended=form_data.get(' SURF_extended_switch'), upright=form_data.get('SURF_upright_switch'))
        keypoints.extend(surf_kp)

    # Use SIFT if specified
    if use_sift:
        sift_kp, _ = compute_sift(image, nfeatures=form_data.get('nfeatures'), nOctaveLayers=form_data.get('nOctaveLayers'), contrastThreshold=form_data.get('contrastThreshold'), edgeThreshold=form_data.get('edgeThreshold'), sigma=form_data.get('sigma'), compute_descriptor=form_data.get('compute_descriptor'))
        keypoints.extend(sift_kp)
    
    # Draw the keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return image_with_keypoints

def combined_matching(img1, img2, form_data, use_orb=True, use_surf=True, use_sift=True, image_matching_type="FLANN"):
    """
    Matches keypoints and descriptors between two images using ORB, SURF, SIFT or any combination.
    
    Parameters:
    - img1, img2: Input images.
    - use_orb: Boolean, whether to use ORB for matching.
    - use_surf: Boolean, whether to use SURF for matching.
    - use_sift: Boolean, whether to use SIFT for matching.
    
    Returns:
    - good_matches: List of good matches after applying ratio test.
    """

    if not use_orb and not use_surf and not use_sift:
        raise ValueError("At least one of 'use_orb', 'use_surf', or 'use_sift' must be True.")
    
    # Lists to collect keypoints and descriptors from all algorithms
    keypoints1, descriptors1 = [], []
    keypoints2, descriptors2 = [], []

    # Use ORB if specified
    if use_orb:
        orb_kp1, orb_desc1 = compute_orb(img1, n_features=form_data.get('ORB_nfeatures_range'), scaleFactor=form_data.get('ORB_scaleFactor_range'), nLevels=form_data.get('ORB_nLevels_range'), edgeThreshold=form_data.get('ORB_edgeThreshold_range'),  firstLevel=form_data.get('ORB_firstLevel_range'), WTA_K=form_data.get('ORB_WTAK_range'), patchSize=form_data.get('ORB_patchSize_range'), fastThreshold=form_data.get('ORB_fastThreshold_range'), scoreType=form_data.get('ORB_scoreType_radio'))
        orb_kp2, orb_desc2 = compute_orb(img2, n_features=form_data.get('ORB_nfeatures_range'), scaleFactor=form_data.get('ORB_scaleFactor_range'), nLevels=form_data.get('ORB_nLevels_range'), edgeThreshold=form_data.get('ORB_edgeThreshold_range'),  firstLevel=form_data.get('ORB_firstLevel_range'), WTA_K=form_data.get('ORB_WTAK_range'), patchSize=form_data.get('ORB_patchSize_range'), fastThreshold=form_data.get('ORB_fastThreshold_range'), scoreType=form_data.get('ORB_scoreType_radio'))
        
        keypoints1.extend(orb_kp1)
        descriptors1.extend(orb_desc1)
        keypoints2.extend(orb_kp2)
        descriptors2.extend(orb_desc2)

    # Use SIFT if specified
    if use_sift:
        sift_kp1, sift_desc1 = compute_sift(img1, nfeatures=form_data.get('nfeatures'), nOctaveLayers=form_data.get('nOctaveLayers'), contrastThreshold=form_data.get('contrastThreshold'), edgeThreshold=form_data.get('edgeThreshold'), sigma=form_data.get('sigma'), compute_descriptor=form_data.get('compute_descriptor'))
        sift_kp2, sift_desc2 = compute_sift(img2, nfeatures=form_data.get('nfeatures'), nOctaveLayers=form_data.get('nOctaveLayers'), contrastThreshold=form_data.get('contrastThreshold'), edgeThreshold=form_data.get('edgeThreshold'), sigma=form_data.get('sigma'), compute_descriptor=form_data.get('compute_descriptor'))

        keypoints1.extend(sift_kp1)
        descriptors1.extend(sift_desc1)
        keypoints2.extend(sift_kp2)
        descriptors2.extend(sift_desc2)

    descriptors1 = np.array(descriptors1)
    descriptors2 = np.array(descriptors2)

    if image_matching_type == "FLANN-radio":
        print("im using flann")
        # Convert descriptors to float32 as required by FLANN-based matcher
        descriptors1 = descriptors1.astype(np.float32)
        descriptors2 = descriptors2.astype(np.float32)

        # Define FLANN parameters and use it to match the descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    else:
        # Create BFMatcher and match the descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test for better quality matches (optional)
    print("Matches: ", matches)
    good_matches = []
    for m, n in matches:
        print(m.distance, n.distance)
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, keypoints1, keypoints2



def compute_surf(image, hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=True, upright=False):
    """
    Compute SURF keypoints and descriptors for a given image.

    Parameters:
    - image: Input image (grayscale).
    - hessianThreshold: Threshold for the keypoint detector. Only features with hessian larger than hessianThreshold are retained.
    - nOctaves: Number of pyramid octaves the keypoint detector will use.
    - nOctaveLayers: Number of octave layers within each octave.
    - extended: If True, computes the extended descriptor (128 elements). Otherwise, computes the basic one (64 elements).
    - upright: If True, doesn't compute the orientation of the keypoint.

    Returns:
    - keypoints: List of detected keypoints.
    - descriptors: SURF descriptors for the detected keypoints.
    """

    # Initialize the SURF detector with the parameters
    surf = cv2.SURF_create(
        hessianThreshold=hessianThreshold,
        nOctaves=nOctaves,
        nOctaveLayers=nOctaveLayers,
        extended=extended,
        upright=upright
    )

    # Compute keypoints and descriptors
    keypoints, descriptors = surf.detectAndCompute(image, None)

    return keypoints, np.array(descriptors)


def compute_orb(image, n_features=500, scaleFactor=1.2, nLevels=8, edgeThreshold=31, 
                firstLevel=0, WTA_K=2, patchSize=31, fastThreshold=20, scoreType=cv2.ORB_HARRIS_SCORE):
    """
    Compute ORB keypoints and descriptors for a given image.

    Parameters:
    - image: The input image on which ORB computation will be performed.
    - n_features: The maximum number of features to retain.
    - scaleFactor: Pyramid decimation ratio.
    - nLevels: The number of pyramid levels.
    - edgeThreshold: Size of the border where the features are not detected.
    - firstLevel: The level of the pyramid to put source image to.
    - WTA_K: The number of points that produce each element of the oriented BRIEF descriptor.
    - patchSize: Size of the patch used by the oriented BRIEF descriptor.
    - fastThreshold: The default threshold to use in the FAST keypoint detector.
    - scoreType: The type of score to use (either cv2.ORB_HARRIS_SCORE or cv2.ORB_FAST_SCORE).

    Returns:
    - keypoints: List of detected keypoints.
    - descriptors: ORB descriptors for the detected keypoints.
    """
    if scoreType == 'FAST_SCORE':
        scoreType = cv2.ORB_FAST_SCORE
    else:
        scoreType = cv2.ORB_HARRIS_SCORE

    # Initialize ORB detector with the given parameters
    orb = cv2.ORB_create(nfeatures=int(n_features), scaleFactor=float(scaleFactor), nlevels=int(nLevels),
                         edgeThreshold=int(edgeThreshold), firstLevel=int(firstLevel), WTA_K=int(WTA_K),
                         patchSize=int(patchSize), fastThreshold=int(fastThreshold), scoreType=int(scoreType))

    # Compute the keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, np.array(descriptors)


def compute_sift(image,
                 nfeatures,
                 nOctaveLayers,
                 contrastThreshold,
                 edgeThreshold,
                 sigma,
                 compute_descriptor):
    """
    Compute SIFT keypoints and descriptors for the input image.
    
    Parameters:
    - image: Input image (grayscale).
    - Other parameters: Parameters for the SIFT detector and descriptor.
    
    Returns:
    - keypoints: List of detected keypoints.
    - descriptors: Numpy array of descriptors. Returned only if compute_descriptor is True.
    """
    # Create the SIFT detector object parameters dictionary
    sift_params = {
        'nfeatures': nfeatures,
        'nOctaveLayers': nOctaveLayers,
        'contrastThreshold': contrastThreshold,
        'edgeThreshold': edgeThreshold,
        'sigma': sigma,
    }

    # Create a SIFT detector object with the specified parameters
    sift = cv2.SIFT_create(**sift_params)
    
    # Compute SIFT keypoints
    keypoints = sift.detect(image, None)
    
    keypoints, descriptors = sift.compute(image, keypoints)
    
    return keypoints, np.array(descriptors)


def compute_transformation(img1, img2, keypoints1, keypoints2, good_matches, form_data):
    """
    Compute the geometric transformation that aligns img2 onto img1.
    
    Parameters:
    - img1, img2: Input images.
    - keypoints1, keypoints2: Detected keypoints from img1 and img2.
    - good_matches: Good matches after applying ratio test.
    - form_data: User input data containing choices for estimation technique and transformation model.

    Returns:
    - M: Transformation matrix.
    - mask: Inliers used for transformation estimation.
    """

    # Get the coordinates of the good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # check if src_pts and dst_pts have same sizes
    print(src_pts.shape, dst_pts.shape)
    if src_pts.shape != dst_pts.shape:    
        raise ValueError("src_pts and dst_pts must have the same size!")
    
    # if good matches are less than 3, then we cannot estimate the transformation
    print(len(good_matches))
    print(len(src_pts), len(dst_pts))
    if len(good_matches) < 3:
        raise ValueError("Need at least 3 good matches to estimate a transformation.")

    # Estimate the transformation
    print(form_data)
    print(form_data.get('model-selection-radio'))
    print(form_data.get('estimation-techniques-radio'))
    if form_data.get('estimation-techniques-radio') == "RANSAC-radio":
        method = cv2.RANSAC
    elif form_data.get('estimation-techniques-radio') == "least-median-squares-radio":
        method = cv2.LMEDS
    else:
        raise ValueError("Invalid estimation technique provided.")
    
    
    if form_data.get('model-selection-radio') == "affine-radio":
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=method)
    elif form_data.get('model-selection-radio') == "projective-radio":
        M, mask = cv2.findHomography(src_pts, dst_pts, method=method)
    elif form_data.get('model-selection-radio') == "rigid-radio":
        # For Rigid transformation, decompose the Affine to get rotation and translation
        affine, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=method)
        rot, trans = cv2.decomposeAffineTransform(affine)
        M = np.vstack([rot, trans])
    else:
        raise ValueError("Invalid transformation model provided.")

    return M, mask


def align_images(img1, img2, M, form_data):
    """
    Aligns img2 onto img1 using the provided transformation matrix.
    
    Parameters:
    - img1: Reference image.
    - img2: Image to be aligned.
    - M: Transformation matrix.
    - form_data: User input data containing choices for interpolation method.

    Returns:
    - aligned_img: The aligned version of img2.
    """

    # Get the size of img1
    h, w = img1.shape[:2]
    
    # Define interpolation method
    if form_data.get('interpolation-radio') == "Bilinear Interpolation":
        interp_method = cv2.INTER_LINEAR
    elif form_data.get('interpolation-radio') == "Bicubic Interpolation":
        interp_method = cv2.INTER_CUBIC
    else:
        raise ValueError("Invalid interpolation method provided.")
    
    # Apply transformation
    if form_data.get('model-selection-radio') == "affine-radio" or form_data.get('ModelSelection') == "Rigid":
        if M is None:
            raise ValueError("The transformation matrix M is None!")
        aligned_img = cv2.warpAffine(img2, M[:2], (w, h), flags=interp_method)
    elif form_data.get('model-selection-radio') == "projective-radio":
        aligned_img = cv2.warpPerspective(img2, M, (w, h), flags=interp_method)
    else:
        raise ValueError("Invalid transformation model provided.")
    
    return aligned_img

def anomaly_detection(request):
    # randomly grab one image
    image = UploadedImage.objects.order_by('?').first()
    print(image.image)
    return render(request, 'anomaly_detection.html', {'image': image})

def output(request):
    # randomly grab one image
    image = UploadedImage.objects.order_by('?').first()
    print(image.image)
    return render(request, 'output.html', {'image': image})