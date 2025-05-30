import pandas as pd
import numpy as np
from collections import defaultdict
import json


# Function to load and process the CSV file
def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert relevant columns to appropriate types
    df['Latency_Seconds'] = pd.to_numeric(df['Latency_Seconds'], errors='coerce')
    df['Cost_USD'] = pd.to_numeric(df['Cost_USD'], errors='coerce')

    return df


# Function to extract unique images and create ground truth objects
def create_ground_truth(df):
    # Get unique image names
    unique_images = df['Image_Name'].unique()
    print(f"Found {len(unique_images)} unique images")

    # Create a ground truth dictionary for each unique image
    ground_truth = {}

    for image in unique_images:
        # Get all rows for this image
        image_data = df[df['Image_Name'] == image]

        # Extract all parsing results for this image across different models
        object_names = image_data['Parsed_Object_Name'].tolist()
        object_shapes = image_data['Parsed_Object_Shape'].tolist()
        object_dimensions = image_data['Parsed_Object_Dimensions_mm'].tolist()
        object_orientations = image_data['Parsed_Object_Orientation'].tolist()
        grasp_types = image_data['Parsed_Grasp_Type'].tolist()
        hand_rotations = image_data['Parsed_Hand_Rotation_Deg'].tolist()
        hand_apertures = image_data['Parsed_Hand_Aperture_mm'].tolist()
        num_fingers = image_data['Parsed_Num_Fingers'].tolist()

        # Standardize dimensions before finding mode
        standardized_dimensions = [standardize_dimensions(dim) for dim in object_dimensions if dim]

        # For object_name, we'll store a list of all parsed names
        # For numeric/categorical attributes, we'll determine the most common value
        gt = {
            "image_name": image,
            "object_names": list(set(name for name in object_names if pd.notna(name))),  # List of unique non-NaN names
            "object_shape": mode([s for s in object_shapes if pd.notna(s)]),  # Most common shape
            "object_dimensions_mm": standardize_dimensions(
                mode(standardized_dimensions)) if standardized_dimensions else None,
            # Most common dimensions (standardized)
            "object_orientation": mode([o for o in object_orientations if pd.notna(o)]),  # Most common orientation
            "grasp_type": mode([gt for gt in grasp_types if pd.notna(gt)]),  # Most common grasp type
            "hand_rotation_deg": mode([hr for hr in hand_rotations if pd.notna(hr)]),  # Most common rotation
            "hand_aperture_mm": mode([ha for ha in hand_apertures if pd.notna(ha)]),  # Most common aperture
            "num_fingers": mode([nf for nf in num_fingers if pd.notna(nf)])  # Most common finger count
        }

        ground_truth[image] = gt

    return ground_truth


# Function to find the most common item in a list
def mode(lst):
    if not lst:
        return None
    # Filter out potential NaN values if they are not hashable or comparable
    filtered_lst = [item for item in lst if pd.notna(item)]
    if not filtered_lst:
        return None
    try:
        return max(set(filtered_lst), key=filtered_lst.count)
    except TypeError:  # Handle unhashable types like lists if they sneak in
        # Fallback for unhashable items: convert to string for mode finding, or handle specific cases
        try:
            str_lst = [str(item) for item in filtered_lst]
            return max(set(str_lst), key=str_lst.count)  # This might not be ideal for all types
        except:
            return None


# Function to standardize dimension format to WHD (Width, Height, Depth)
# Where: W = second-longest, H = longest, D = shortest
def standardize_dimensions(dim_str):
    if pd.isna(dim_str):
        return None
    try:
        # Parse the dimensions
        dims = [int(d) for d in str(dim_str).split('x')]
        if len(dims) != 3:
            return str(dim_str)  # Return original if not 3 dimensions

        # Sort dimensions (ascending)
        sorted_dims = sorted(dims)

        # Rearrange according to WHD convention
        # D = shortest, H = longest, W = middle/second-longest
        d = sorted_dims[0]  # shortest
        w = sorted_dims[1]  # middle/second-longest
        h = sorted_dims[2]  # longest

        # Return in WHD format
        return f"{w}x{h}x{d}"
    except:
        return str(dim_str)  # Return original if parsing fails


# Function to calculate error metrics for numeric attributes
def calculate_numeric_metrics(df, ground_truth):
    print("Calculating numeric metrics...")

    # Initialize metrics storage for lists of errors
    numeric_attributes = ['Parsed_Object_Dimensions_mm', 'Parsed_Hand_Rotation_Deg', 'Parsed_Hand_Aperture_mm',
                          'Parsed_Num_Fingers']
    # These will store lists of errors for each model/dimension
    # 'MAE_errors' will store absolute errors, 'ME_errors' will store raw errors
    error_collections = {}
    for attribute in numeric_attributes:
        error_collections[attribute] = {
            'absolute_errors': defaultdict(list),
            'raw_errors': defaultdict(list)
        }

    def process_dimensions(dim_str):
        if pd.isna(dim_str):
            return None
        try:
            std_dim_str = standardize_dimensions(str(dim_str))
            if std_dim_str is None: return None  # If standardization returns None
            dims = [int(d) for d in std_dim_str.split('x')]
            return dims
        except:
            return None

    def calculate_min_error(pred_value, gt_values):
        if pd.isna(pred_value):
            return None, None

        pred_value_float = float(pred_value)

        if isinstance(gt_values, list):
            # Ensure all gt_values are float for comparison
            gt_values_float = [float(gt_val) for gt_val in gt_values if pd.notna(gt_val)]
            if not gt_values_float: return None, None

            abs_errors = [abs(pred_value_float - gt_val_f) for gt_val_f in gt_values_float]
            errors = [pred_value_float - gt_val_f for gt_val_f in gt_values_float]

            min_abs_error = min(abs_errors)
            min_idx = abs_errors.index(min_abs_error)
            return min_abs_error, errors[min_idx]
        else:
            if pd.isna(gt_values): return None, None
            gt_value_float = float(gt_values)
            abs_error = abs(pred_value_float - gt_value_float)
            error = pred_value_float - gt_value_float
            return abs_error, error

    for idx, row in df.iterrows():
        image = row['Image_Name']
        model = row['VLM_Model']

        if image not in ground_truth:
            continue
        gt = ground_truth[image]

        # Dimensions
        if pd.notna(row['Parsed_Object_Dimensions_mm']) and gt['object_dimensions_mm'] is not None:
            pred_dims = process_dimensions(row['Parsed_Object_Dimensions_mm'])
            gt_dims = process_dimensions(gt['object_dimensions_mm'])

            if pred_dims and gt_dims and len(pred_dims) == len(gt_dims):
                for i, (p, g) in enumerate(zip(pred_dims, gt_dims)):
                    abs_err = abs(p - g)
                    raw_err = p - g
                    error_collections['Parsed_Object_Dimensions_mm']['absolute_errors'][f"{model}_dim{i + 1}"].append(
                        abs_err)
                    error_collections['Parsed_Object_Dimensions_mm']['raw_errors'][f"{model}_dim{i + 1}"].append(
                        raw_err)

        # Rotation
        if pd.notna(row['Parsed_Hand_Rotation_Deg']) and gt['hand_rotation_deg'] is not None:
            abs_error, error = calculate_min_error(row['Parsed_Hand_Rotation_Deg'], gt['hand_rotation_deg'])
            if abs_error is not None and error is not None:
                error_collections['Parsed_Hand_Rotation_Deg']['absolute_errors'][model].append(abs_error)
                error_collections['Parsed_Hand_Rotation_Deg']['raw_errors'][model].append(error)

        # Aperture
        if pd.notna(row['Parsed_Hand_Aperture_mm']) and gt['hand_aperture_mm'] is not None:
            abs_error, error = calculate_min_error(row['Parsed_Hand_Aperture_mm'], gt['hand_aperture_mm'])
            if abs_error is not None and error is not None:
                error_collections['Parsed_Hand_Aperture_mm']['absolute_errors'][model].append(abs_error)
                error_collections['Parsed_Hand_Aperture_mm']['raw_errors'][model].append(error)

        # Finger count
        if pd.notna(row['Parsed_Num_Fingers']) and gt['num_fingers'] is not None:
            abs_error, error = calculate_min_error(row['Parsed_Num_Fingers'], gt['num_fingers'])
            if abs_error is not None and error is not None:
                error_collections['Parsed_Num_Fingers']['absolute_errors'][model].append(abs_error)
                error_collections['Parsed_Num_Fingers']['raw_errors'][model].append(error)

    # Compute final MAE, ME, MAE_STD, ME_STD
    results = {}
    for attribute in numeric_attributes:
        results[attribute] = {
            'MAE': {}, 'MAE_STD': {},
            'ME': {}, 'ME_STD': {}
        }
        # For MAE (from absolute errors)
        for model_key, abs_error_list in error_collections[attribute]['absolute_errors'].items():
            if abs_error_list:
                results[attribute]['MAE'][model_key] = np.mean(abs_error_list)
                results[attribute]['MAE_STD'][model_key] = np.std(abs_error_list)
            else:  # Handle cases with no valid errors
                results[attribute]['MAE'][model_key] = np.nan
                results[attribute]['MAE_STD'][model_key] = np.nan

        # For ME (from raw errors)
        for model_key, raw_error_list in error_collections[attribute]['raw_errors'].items():
            if raw_error_list:
                results[attribute]['ME'][model_key] = np.mean(raw_error_list)
                results[attribute]['ME_STD'][model_key] = np.std(raw_error_list)
            else:  # Handle cases with no valid errors
                results[attribute]['ME'][model_key] = np.nan
                results[attribute]['ME_STD'][model_key] = np.nan

    return results


# Function to check if a value is in a ground truth, which may be a single value or a list
def is_in_ground_truth(pred_value, gt_value):
    if pd.isna(pred_value):
        return False
    if pd.isna(gt_value):  # If ground truth is NaN, can't be correct unless pred is also NaN (handled above)
        return False

    # If ground truth is a list, check if prediction is in the list
    if isinstance(gt_value, list):
        # Convert all values to same type for comparison
        try:
            if isinstance(pred_value, (int, float, np.number)):
                gt_value_numeric = [float(v) for v in gt_value if
                                    pd.notna(v) and isinstance(v, (int, float, np.number))]
                return float(pred_value) in gt_value_numeric
            else:  # string comparison
                gt_value_str = [str(v) for v in gt_value if pd.notna(v)]
                return str(pred_value) in gt_value_str
        except ValueError:  # If conversion fails
            return str(pred_value) in [str(v) for v in gt_value if pd.notna(v)]  # Fallback to string
    # Otherwise, direct comparison
    else:
        try:
            # Convert to same type for comparison if both are numeric-like
            if isinstance(pred_value, (int, float, np.number)) and isinstance(gt_value, (int, float, np.number)):
                return float(pred_value) == float(gt_value)
            return str(pred_value) == str(gt_value)  # Fallback to string comparison
        except ValueError:
            return str(pred_value) == str(gt_value)


# Function to calculate categorical accuracy
def calculate_categorical_metrics(df, ground_truth):
    print("Calculating categorical metrics...")

    # Initialize metrics
    categorical_attributes = ['Parsed_Object_Shape', 'Parsed_Object_Orientation', 'Parsed_Grasp_Type']
    metrics = {}

    for attribute in categorical_attributes:
        metrics[attribute] = defaultdict(list)  # Accuracy by model

    # Create a special metric for object name
    metrics['Parsed_Object_Name'] = defaultdict(list)

    # Process each row
    for idx, row in df.iterrows():
        image = row['Image_Name']
        model = row['VLM_Model']

        if image not in ground_truth:
            continue

        gt = ground_truth[image]

        # Handle object name specially
        if pd.notna(row['Parsed_Object_Name']) and gt['object_names'] is not None:
            pred_name = str(row['Parsed_Object_Name'])  # ensure string
            # GT names are already list of strings
            is_correct = pred_name in gt['object_names']
            metrics['Parsed_Object_Name'][model].append(is_correct)

        # Handle shape
        if pd.notna(row['Parsed_Object_Shape']) and gt['object_shape'] is not None:
            pred_shape = str(row['Parsed_Object_Shape'])
            is_correct = is_in_ground_truth(pred_shape, gt['object_shape'])
            metrics['Parsed_Object_Shape'][model].append(is_correct)

        # Handle orientation
        if pd.notna(row['Parsed_Object_Orientation']) and gt['object_orientation'] is not None:
            pred_orient = str(row['Parsed_Object_Orientation'])
            is_correct = is_in_ground_truth(pred_orient, gt['object_orientation'])
            metrics['Parsed_Object_Orientation'][model].append(is_correct)

        # Handle grasp type
        if pd.notna(row['Parsed_Grasp_Type']) and gt['grasp_type'] is not None:
            pred_grasp = row['Parsed_Grasp_Type']  # Can be int or str from CSV
            # GT grasp type can be int or list of ints
            is_correct = is_in_ground_truth(pred_grasp, gt['grasp_type'])
            metrics['Parsed_Grasp_Type'][model].append(is_correct)

    # Compute accuracy
    results = {}
    for attribute in categorical_attributes + ['Parsed_Object_Name']:
        results[attribute] = {}
        for model in metrics[attribute]:
            if metrics[attribute][model]:
                results[attribute][model] = np.mean(metrics[attribute][model])
            else:
                results[attribute][model] = np.nan  # If no data for this model/attribute combo

    return results


# Function to analyze dataset statistics
# Function to analyze dataset statistics
def analyze_dataset_stats(df, ground_truth):
    print("Analyzing dataset statistics...")

    stats = {}

    # Basic dataset stats
    stats['total_images'] = len(ground_truth)
    stats['total_samples'] = len(df)
    stats['unique_models'] = len(df['VLM_Model'].unique())
    stats['models'] = sorted(df['VLM_Model'].unique().tolist())

    # Count object shapes
    shape_counts = defaultdict(int)
    for image_data in ground_truth.values():
        shape = image_data.get('object_shape')
        if pd.notna(shape):  # Shape is expected to be scalar or None/NaN
            shape_counts[str(shape)] += 1
    stats['shape_counts'] = dict(shape_counts)

    # Count orientations
    orientation_counts = defaultdict(int)
    for image_data in ground_truth.values():
        orientation = image_data.get('object_orientation')
        if pd.notna(orientation):  # Orientation is expected to be scalar or None/NaN
            orientation_counts[str(orientation)] += 1
    stats['orientation_counts'] = dict(orientation_counts)

    # Count grasp types
    grasp_counts = defaultdict(int)
    for image_data in ground_truth.values():
        grasp_val = image_data.get('grasp_type')

        if grasp_val is None:  # Check if the entire field is missing
            continue

        if isinstance(grasp_val, list):
            for g_item in grasp_val:
                if pd.notna(g_item):  # Check individual items in the list
                    grasp_counts[f"type_{g_item}"] += 1
        else:  # It's a scalar value
            if pd.notna(grasp_val):  # Check the scalar value
                grasp_counts[f"type_{grasp_val}"] += 1
    stats['grasp_counts'] = dict(grasp_counts)

    # Count rotation angles
    rotation_counts = defaultdict(int)
    for image_data in ground_truth.values():
        rotation_val = image_data.get('hand_rotation_deg')

        if rotation_val is None:  # Check if the entire field is missing
            continue

        if isinstance(rotation_val, list):
            for r_item in rotation_val:
                if pd.notna(r_item):
                    r_key = f"{int(float(r_item))}_deg" if float(r_item).is_integer() else f"{float(r_item)}_deg"
                    rotation_counts[r_key] += 1
        else:
            if pd.notna(rotation_val):
                r_key = f"{int(float(rotation_val))}_deg" if float(
                    rotation_val).is_integer() else f"{float(rotation_val)}_deg"
                rotation_counts[r_key] += 1
    stats['rotation_counts'] = dict(rotation_counts)

    # Count finger configurations
    finger_counts = defaultdict(int)
    for image_data in ground_truth.values():
        fingers_val = image_data.get('num_fingers')

        if fingers_val is None:  # Check if the entire field is missing
            continue

        if isinstance(fingers_val, list):
            for f_item in fingers_val:
                if pd.notna(f_item):  # Check individual items in the list
                    finger_counts[f"{f_item}_fingers"] += 1
        else:  # It's a scalar value
            if pd.notna(fingers_val):  # Check the scalar value
                finger_counts[f"{fingers_val}_fingers"] += 1
    stats['finger_counts'] = dict(finger_counts)

    return stats

# Function to analyze latency and cost by vendor
def analyze_latency_cost(df):
    print("Analyzing latency and cost...")

    # Group by model
    model_metrics = {}

    models = df['VLM_Model'].unique()
    for model in models:
        model_data = df[df['VLM_Model'] == model]

        model_metrics[model] = {
            'avg_latency': model_data['Latency_Seconds'].mean(),
            'std_latency': model_data['Latency_Seconds'].std(),  # Added STD for latency
            'min_latency': model_data['Latency_Seconds'].min(),
            'max_latency': model_data['Latency_Seconds'].max(),
            'avg_cost': model_data['Cost_USD'].mean(),
            'total_cost': model_data['Cost_USD'].sum(),
            'count': len(model_data)
        }

    return model_metrics


# Function to load ground truth from file if it exists
def load_ground_truth(file_path="ground_truth.json"):
    try:
        print(f"Attempting to load ground truth from {file_path}...")
        with open(file_path, 'r') as f:
            gt_data = json.load(f)
            # Convert numeric strings back to numbers if needed, especially for lists
            for img_name, data in gt_data.items():
                if 'hand_rotation_deg' in data and isinstance(data['hand_rotation_deg'], list):
                    data['hand_rotation_deg'] = [pd.to_numeric(x, errors='coerce') for x in data['hand_rotation_deg']]
                elif 'hand_rotation_deg' in data:
                    data['hand_rotation_deg'] = pd.to_numeric(data['hand_rotation_deg'], errors='coerce')

                if 'grasp_type' in data and isinstance(data['grasp_type'], list):
                    data['grasp_type'] = [pd.to_numeric(x, errors='coerce') for x in data['grasp_type']]
                elif 'grasp_type' in data:
                    data['grasp_type'] = pd.to_numeric(data['grasp_type'], errors='coerce')

                if 'num_fingers' in data and isinstance(data['num_fingers'], list):
                    data['num_fingers'] = [pd.to_numeric(x, errors='coerce') for x in data['num_fingers']]
                elif 'num_fingers' in data:
                    data['num_fingers'] = pd.to_numeric(data['num_fingers'], errors='coerce')
            return gt_data
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Ground truth file not found or invalid. Will create new ground truth.")
        return None


# Main function
def main():
    csv_path = "vlm_benchmark_results-may25.csv"  # Make sure this file exists
    df = load_data(csv_path)

    # Try to load existing ground truth, create new one if not available
    ground_truth = load_ground_truth()
    if ground_truth is None:
        print("Creating new ground truth from data...")
        ground_truth = create_ground_truth(df)
        # Save the newly created ground truth
        with open('ground_truth.json', 'w') as f:
            json.dump(ground_truth, f, indent=2)
        print("New ground truth saved to ground_truth.json")
    else:
        print("Using existing ground truth file.")

    # Calculate metrics and dataset statistics
    dataset_stats = analyze_dataset_stats(df, ground_truth)
    numeric_metrics = calculate_numeric_metrics(df, ground_truth)
    categorical_metrics = calculate_categorical_metrics(df, ground_truth)
    latency_cost_metrics = analyze_latency_cost(df)

    # Create summary report
    print("\n===== DATASET STATISTICS =====\n")

    # Print basic dataset stats
    print(f"Total unique images: {dataset_stats['total_images']}")
    print(f"Total samples in dataset: {dataset_stats['total_samples']}")
    print(f"Number of VLM models evaluated: {dataset_stats['unique_models']}")
    print(f"Models: {', '.join(dataset_stats['models'])}")

    # Print object shape distribution
    print("\nOBJECT SHAPE DISTRIBUTION:")
    for shape, count in sorted(dataset_stats['shape_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {shape}: {count} objects ({count / dataset_stats['total_images']:.1%})")

    # Print orientation distribution
    print("\nORIENTATION DISTRIBUTION:")
    for orientation, count in sorted(dataset_stats['orientation_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {orientation}: {count} objects ({count / dataset_stats['total_images']:.1%})")

    # Print grasp type distribution
    print("\nGRASP TYPE DISTRIBUTION:")
    for grasp, count in sorted(dataset_stats['grasp_counts'].items(), key=lambda x: x[1], reverse=True):
        # Calculate percentage based on total images that *have* a grasp type specified
        total_grasps_specified = sum(dataset_stats['grasp_counts'].values())
        percentage = (count / total_grasps_specified * 100) if total_grasps_specified > 0 else 0
        print(f"  {grasp}: {count} instances ({percentage:.1f}% of specified)")

    # Print rotation angle distribution
    print("\nROTATION ANGLE DISTRIBUTION:")
    for rotation, count in sorted(dataset_stats['rotation_counts'].items(), key=lambda x: x[1], reverse=True):
        total_rotations_specified = sum(dataset_stats['rotation_counts'].values())
        percentage = (count / total_rotations_specified * 100) if total_rotations_specified > 0 else 0
        print(f"  {rotation}: {count} instances ({percentage:.1f}% of specified)")

    # Print finger count distribution
    print("\nFINGER COUNT DISTRIBUTION:")
    for fingers, count in sorted(dataset_stats['finger_counts'].items(), key=lambda x: x[1], reverse=True):
        total_fingers_specified = sum(dataset_stats['finger_counts'].values())
        percentage = (count / total_fingers_specified * 100) if total_fingers_specified > 0 else 0
        print(f"  {fingers}: {count} instances ({percentage:.1f}% of specified)")

    print("\n===== MODEL PERFORMANCE METRICS =====\n")

    # Print model performance by category
    print("CATEGORICAL ACCURACY BY MODEL:")
    for attribute in categorical_metrics:
        print(f"\n{attribute.replace('Parsed_', '')}:")
        # Sort models by accuracy (descending), handling potential NaN for missing data
        sorted_models = sorted(
            categorical_metrics[attribute].items(),
            key=lambda x: (isinstance(x[1], float) and not np.isnan(x[1]),
                           x[1] if isinstance(x[1], float) and not np.isnan(x[1]) else -1),
            reverse=True
        )
        for model, accuracy in sorted_models:
            if pd.notna(accuracy):
                print(f"  {model}: {accuracy:.2%}")
            else:
                print(f"  {model}: N/A (no data)")

    # Print numeric metrics (MAE ± STD)
    print("\nNUMERIC METRICS (MAE ± STD):")
    for attribute in numeric_metrics:
        print(f"\n{attribute.replace('Parsed_', '')}:")
        maes = numeric_metrics[attribute]['MAE']
        maes_std = numeric_metrics[attribute]['MAE_STD']

        # Sort by MAE value (ascending), handling NaNs
        sorted_models = sorted(
            maes.keys(),
            key=lambda model_key: (
            pd.notna(maes[model_key]), maes[model_key] if pd.notna(maes[model_key]) else float('inf'))
        )
        for model_key in sorted_models:
            mae_val = maes[model_key]
            std_val = maes_std.get(model_key, np.nan)
            if pd.notna(mae_val):
                print(f"  {model_key}: {mae_val:.2f} ± {std_val:.2f}")
            else:
                print(f"  {model_key}: N/A ± N/A")

    # Print numeric metrics (ME ± STD to show bias)
    print("\nNUMERIC METRICS (ME ± STD - negative means underestimation, positive means overestimation):")
    for attribute in numeric_metrics:
        print(f"\n{attribute.replace('Parsed_', '')}:")
        mes = numeric_metrics[attribute]['ME']
        mes_std = numeric_metrics[attribute]['ME_STD']

        # Sort by absolute ME value (ascending), handling NaNs
        sorted_models = sorted(
            mes.keys(),
            key=lambda model_key: (
            pd.notna(mes[model_key]), abs(mes[model_key]) if pd.notna(mes[model_key]) else float('inf'))
        )
        for model_key in sorted_models:
            me_val = mes[model_key]
            std_val = mes_std.get(model_key, np.nan)
            if pd.notna(me_val):
                print(f"  {model_key}: {me_val:.2f} ± {std_val:.2f}")
            else:
                print(f"  {model_key}: N/A ± N/A")

    # Print latency and cost
    print("\nLATENCY AND COST BY MODEL (Avg Latency ± STD):")  # Added STD to latency printout
    # Sort by average latency
    sorted_latency_cost = sorted(latency_cost_metrics.items(), key=lambda x: (
    pd.notna(x[1]['avg_latency']), x[1]['avg_latency'] if pd.notna(x[1]['avg_latency']) else float('inf')))

    for model, metrics_val in sorted_latency_cost:
        print(f"\n{model}:")
        if pd.notna(metrics_val['avg_latency']):
            print(
                f"  Avg Latency: {metrics_val['avg_latency']:.2f} ± {metrics_val.get('std_latency', np.nan):.2f} seconds")
            print(f"  Latency Range: {metrics_val['min_latency']:.2f} - {metrics_val['max_latency']:.2f} seconds")
        else:
            print(f"  Avg Latency: N/A")
            print(f"  Latency Range: N/A - N/A")

        if pd.notna(metrics_val['avg_cost']):
            print(f"  Avg Cost: ${metrics_val['avg_cost']:.6f}")
            print(f"  Total Cost: ${metrics_val['total_cost']:.6f}")
        else:
            print(f"  Avg Cost: N/A")
            print(f"  Total Cost: N/A")
        print(f"  Sample Count: {metrics_val['count']}")

    # Save metrics and stats to file
    with open('metrics_results.json', 'w') as f:
        json.dump({
            'dataset_stats': dataset_stats,
            'numeric_metrics': numeric_metrics,
            'categorical_metrics': categorical_metrics,
            'latency_cost_metrics': latency_cost_metrics
        }, f, indent=2, cls=NpEncoder)  # Use NpEncoder for NaN handling

    print("\nResults saved to ground_truth.json and metrics_results.json")


# Custom JSON encoder to handle numpy types like np.nan
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):  # Handle pandas NaN
            return None  # Represent NaN as null in JSON
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    main()