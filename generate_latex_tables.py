#!/usr/bin/env python3
"""
Generate LaTeX tables from metrics_results.json for the paper.
This will return the exact LaTeX code, the only change is "Flash" is shortened in the paper.
"""

import json
from typing import Dict, List, Any


def load_metrics(filename: str) -> dict:
    """Load metrics from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def get_model_display_name(model_id: str) -> str:
    """Convert model ID to display name for the table."""
    name_map = {
        'mistralai/mistral-medium-3': 'Mistral Medium',
        'openai/gpt-4.1': 'GPT-4.1',
        'google/gemini-2.5-flash-preview': 'Gemini 2.5 Flash 17-04',
        'anthropic/claude-3.5-sonnet': 'Claude 3.5 Sonnet',
        'anthropic/claude-3.7-sonnet': 'Claude 3.7 Sonnet',
        'google/gemini-2.5-flash-preview-05-20': 'Gemini 2.5 Flash 20-05',
        'google/gemma-3-27b-it': 'Gemma 3 27B-IT',
        'anthropic/claude-sonnet-4': 'Claude Sonnet 4'
    }
    return name_map.get(model_id, model_id)


def format_percentage(value: float) -> str:
    """Format a value as percentage with 1 decimal place."""
    if value is None or value != value:  # handles None and NaN
        return "N/A"
    return f"{value * 100:.1f}"


def format_float(value: float, decimals: int = 2) -> str:
    """Format a float with specified decimal places."""
    if value is None or value != value:  # handles None and NaN
        return "N/A"
    return f"{value:.{decimals}f}"


def format_mae_std(mae: float, std: float, decimals: int = 2) -> str:
    """Format MAE ± STD string."""
    if mae is None or mae != mae or std is None or std != std:
        return "N/A ± N/A"
    return f"{format_float(mae, decimals)} ± {format_float(std, decimals)}"


def format_me_std(me: float, std: float, decimals: int = 2) -> str:
    """Format ME ± STD string."""
    if me is None or me != me or std is None or std != std:
        return "N/A ± N/A"
    return f"{format_float(me, decimals)} ± {format_float(std, decimals)}"


def sort_models_by_metric(models: List[str], metrics_dict: Dict[str, Any], metric_key: str, reverse: bool = False) -> \
List[str]:
    """Sort models by a specific metric, accessing a nested key if needed."""

    def get_sort_value(model_id):
        model_metric = metrics_dict.get(model_id)
        if isinstance(model_metric, dict):
            val = model_metric.get(metric_key)
        elif isinstance(model_metric, (float, int)):  # Direct value if not a dict
            val = model_metric
        else:  # If model_dims[model] gives a list, like in generate_dimensions_mae_table
            if isinstance(metric_key, int) and isinstance(model_metric, list) and len(model_metric) > metric_key:
                # Assuming model_metric is a list of [ (mae,std), (mae,std), ... ]
                # And metric_key is an index into this list, then access mae.
                if isinstance(model_metric[metric_key], tuple):  # (value, std)
                    val = model_metric[metric_key][0]
                else:  # direct value
                    val = model_metric[metric_key]
            else:
                val = None

        if val is None or val != val:  # Handle None or NaN
            return float('inf') if not reverse else float('-inf')
        return val

    return sorted(models, key=get_sort_value, reverse=reverse)


def generate_categorical_table(data: dict) -> str:
    """Generate the categorical accuracies table."""
    cat_metrics = data['categorical_metrics']
    all_models_ids = data['dataset_stats']['models']

    avg_accuracies = {}
    for model_id in all_models_ids:
        total = 0
        count = 0
        for metric_name in ['Parsed_Object_Name', 'Parsed_Object_Shape', 'Parsed_Object_Orientation',
                            'Parsed_Grasp_Type']:
            val = cat_metrics.get(metric_name, {}).get(model_id)
            if val is not None and val == val:  # not None and not NaN
                total += val
                count += 1
        avg_accuracies[model_id] = total / count if count > 0 else (float('-inf'))  # sort N/A last

    # Sort models by average accuracy across all categorical metrics
    sorted_model_ids = sort_models_by_metric(all_models_ids, avg_accuracies, metric_key=None, reverse=True)

    table = []
    table.append("\\begin{table}[t]")
    table.append("\\caption{Categorical accuracies (\\%); the best results highlighted in \\textbf{bold}.}")
    table.append("\\begin{center}")
    table.append("\\begin{tabular}{|l|c|c|c|c|}")
    table.append("\\hline")
    table.append("\\textbf{Model} & \\textbf{Naming} & \\textbf{Shape} & \\textbf{Orientation} & \\textbf{Grasp type} \\\\")
    table.append("\\hline")

    # Find best values for each metric
    best_naming = max(v for v in cat_metrics['Parsed_Object_Name'].values() if v is not None and v == v)
    best_shape = max(v for v in cat_metrics['Parsed_Object_Shape'].values() if v is not None and v == v)
    best_orientation = max(v for v in cat_metrics['Parsed_Object_Orientation'].values() if v is not None and v == v)
    best_grasp = max(v for v in cat_metrics['Parsed_Grasp_Type'].values() if v is not None and v == v)

    for model_id in sorted_model_ids:
        name_display = get_model_display_name(model_id)
        naming = cat_metrics['Parsed_Object_Name'].get(model_id)
        shape = cat_metrics['Parsed_Object_Shape'].get(model_id)
        orientation = cat_metrics['Parsed_Object_Orientation'].get(model_id)
        grasp = cat_metrics['Parsed_Grasp_Type'].get(model_id)

        naming_str = f"\\textbf{{{format_percentage(naming)}}}" if naming == best_naming and naming is not None else format_percentage(
            naming)
        shape_str = f"\\textbf{{{format_percentage(shape)}}}" if shape == best_shape and shape is not None else format_percentage(
            shape)
        orientation_str = f"\\textbf{{{format_percentage(orientation)}}}" if orientation == best_orientation and orientation is not None else format_percentage(
            orientation)
        grasp_str = f"\\textbf{{{format_percentage(grasp)}}}" if grasp == best_grasp and grasp is not None else format_percentage(
            grasp)

        table.append(f"{name_display:<35} & {naming_str} & {shape_str} & {orientation_str} & {grasp_str} \\\\")

    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\label{tab:cat}")
    table.append("\\end{center}")
    table.append("\\end{table}")

    return '\n'.join(table)


def generate_dimensions_mae_table(data: dict) -> str:
    """Generate the object dimensions MAE table with STD."""
    mae_data = data['numeric_metrics']['Parsed_Object_Dimensions_mm']['MAE']
    std_data = data['numeric_metrics']['Parsed_Object_Dimensions_mm']['MAE_STD']
    all_models_ids = data['dataset_stats']['models']

    model_dims = {}
    for model_id in all_models_ids:
        dims_mae_std = []
        valid_model = True
        for i, dim_label in enumerate(['dim1', 'dim2', 'dim3']):
            key = f"{model_id}_{dim_label}"
            mae_val = mae_data.get(key)
            std_val = std_data.get(key)
            if mae_val is None or std_val is None:
                valid_model = False
                break
            dims_mae_std.append((mae_val, std_val))
        if valid_model and len(dims_mae_std) == 3:
            model_dims[model_id] = dims_mae_std  # Stores list of (mae, std) tuples

    avg_mae_for_sort = {m: sum(dim_tuple[0] for dim_tuple in dims) / 3.0 for m, dims in model_dims.items()}
    sorted_model_ids = sort_models_by_metric(list(model_dims.keys()), avg_mae_for_sort, metric_key=None)

    table = []
    table.append("\\begin{table}[t]")
    table.append("\\caption{MAE ± STD for object dimensions (mm); the lowest MAE highlighted in \\textbf{bold}.}")
    table.append("\\begin{center}")
    table.append("\\begin{tabular}{|l|c|c|c|}")
    table.append("\\hline")
    table.append("\\textbf{Model} & \\textbf{Width} & \\textbf{Height} & \\textbf{Depth} \\\\")  # dim1=Width, dim2=Height, dim3=Depth (as per user's prompt)
    table.append("\\hline")

    # Find best (lowest) MAE values for bolding
    best_width_mae = min(
        dims[0][0] for dims in model_dims.values() if dims[0][0] is not None and dims[0][0] == dims[0][0])
    best_height_mae = min(
        dims[1][0] for dims in model_dims.values() if dims[1][0] is not None and dims[1][0] == dims[1][0])
    best_depth_mae = min(
        dims[2][0] for dims in model_dims.values() if dims[2][0] is not None and dims[2][0] == dims[2][0])

    for model_id in sorted_model_ids:
        name_display = get_model_display_name(model_id)
        width_mae, width_std = model_dims[model_id][0]
        height_mae, height_std = model_dims[model_id][1]
        depth_mae, depth_std = model_dims[model_id][2]

        width_str = format_mae_std(width_mae, width_std)
        height_str = format_mae_std(height_mae, height_std)
        depth_str = format_mae_std(depth_mae, depth_std)

        if width_mae == best_width_mae: width_str = f"\\textbf{{{width_str}}}"
        if height_mae == best_height_mae: height_str = f"\\textbf{{{height_str}}}"
        if depth_mae == best_depth_mae: depth_str = f"\\textbf{{{depth_str}}}"

        table.append(f"{name_display:<35} & {width_str} & {height_str} & {depth_str} \\\\")

    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\label{tab:dim}")
    table.append("\\end{center}")
    table.append("\\end{table}")

    return '\n'.join(table)


def generate_grasp_mae_table(data: dict) -> str:
    """Generate the grasp parameters MAE table with STD."""
    rot_mae = data['numeric_metrics']['Parsed_Hand_Rotation_Deg']['MAE']
    rot_std = data['numeric_metrics']['Parsed_Hand_Rotation_Deg']['MAE_STD']
    ap_mae = data['numeric_metrics']['Parsed_Hand_Aperture_mm']['MAE']
    ap_std = data['numeric_metrics']['Parsed_Hand_Aperture_mm']['MAE_STD']
    fing_mae = data['numeric_metrics']['Parsed_Num_Fingers']['MAE']
    fing_std = data['numeric_metrics']['Parsed_Num_Fingers']['MAE_STD']

    all_models_ids = data['dataset_stats']['models']

    model_metrics_data = {}
    for model_id in all_models_ids:
        r_mae, r_std = rot_mae.get(model_id), rot_std.get(model_id)
        a_mae, a_std = ap_mae.get(model_id), ap_std.get(model_id)
        f_mae, f_std = fing_mae.get(model_id), fing_std.get(model_id)
        if all(v is not None and v == v for v in [r_mae, r_std, a_mae, a_std, f_mae, f_std]):
            model_metrics_data[model_id] = {
                'rot': (r_mae, r_std), 'ap': (a_mae, a_std), 'fing': (f_mae, f_std)
            }

    avg_mae_for_sort = {m: (vals['rot'][0] + vals['ap'][0] + vals['fing'][0]) / 3.0
                        for m, vals in model_metrics_data.items()}
    sorted_model_ids = sort_models_by_metric(list(model_metrics_data.keys()), avg_mae_for_sort, metric_key=None)

    table = []
    table.append("\\begin{table}[t]")
    table.append("\\caption{MAE ± STD for grasp parameters; the lowest MAE highlighted in \\textbf{bold}.}")
    table.append("\\begin{center}")
    table.append("\\begin{tabular}{|l|c|c|c|}")
    table.append("\\hline")
    table.append("\\textbf{Model} & \\textbf{Rotation (°)} & \\textbf{Aperture (mm)} & \\textbf{Fingers} \\\\")
    table.append("\\hline")

    best_rot_mae = min(v['rot'][0] for v in model_metrics_data.values())
    best_ap_mae = min(v['ap'][0] for v in model_metrics_data.values())
    best_fing_mae = min(v['fing'][0] for v in model_metrics_data.values())

    for model_id in sorted_model_ids:
        name_display = get_model_display_name(model_id)
        metrics = model_metrics_data[model_id]

        rot_str = format_mae_std(metrics['rot'][0], metrics['rot'][1], 1)
        ap_str = format_mae_std(metrics['ap'][0], metrics['ap'][1], 2)
        fing_str = format_mae_std(metrics['fing'][0], metrics['fing'][1], 2)

        if metrics['rot'][0] == best_rot_mae: rot_str = f"\\textbf{{{rot_str}}}"
        if metrics['ap'][0] == best_ap_mae: ap_str = f"\\textbf{{{ap_str}}}"
        if metrics['fing'][0] == best_fing_mae: fing_str = f"\\textbf{{{fing_str}}}"

        table.append(f"{name_display:<35} & {rot_str} & {ap_str} & {fing_str} \\\\")

    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\label{tab:grasp}")
    table.append("\\end{center}")
    table.append("\\end{table}")

    return '\n'.join(table)


def generate_dimensions_me_table(data: dict) -> str:
    """Generate the object dimensions ME (bias) table with STD."""
    me_data = data['numeric_metrics']['Parsed_Object_Dimensions_mm']['ME']
    std_data = data['numeric_metrics']['Parsed_Object_Dimensions_mm']['ME_STD']
    all_models_ids = data['dataset_stats']['models']

    model_dims = {}
    for model_id in all_models_ids:
        dims_me_std = []
        valid_model = True
        for dim_label in ['dim1', 'dim2', 'dim3']:
            key = f"{model_id}_{dim_label}"
            me_val = me_data.get(key)
            std_val = std_data.get(key)
            if me_val is None or std_val is None:
                valid_model = False;
                break
            dims_me_std.append((me_val, std_val))
        if valid_model and len(dims_me_std) == 3:
            model_dims[model_id] = dims_me_std

    avg_abs_me_for_sort = {m: sum(abs(dim_tuple[0]) for dim_tuple in dims) / 3.0
                           for m, dims in model_dims.items()}
    sorted_model_ids = sort_models_by_metric(list(model_dims.keys()), avg_abs_me_for_sort, metric_key=None)

    table = []
    table.append("\\begin{table}[t]")
    table.append(
        "\\caption{ME ± STD for object dimensions (mm); ME closest to 0 highlighted in \\textbf{bold}.}")
    table.append("\\begin{center}")
    table.append("\\begin{tabular}{|l|c|c|c|}")
    table.append("\\hline")
    table.append("\\textbf{Model} & \\textbf{Width} & \\textbf{Height} & \\textbf{Depth} \\\\")
    table.append("\\hline")

    closest_width_me = min(
        (dims[0][0] for dims in model_dims.values() if dims[0][0] is not None and dims[0][0] == dims[0][0]), key=abs)
    closest_height_me = min(
        (dims[1][0] for dims in model_dims.values() if dims[1][0] is not None and dims[1][0] == dims[1][0]), key=abs)
    closest_depth_me = min(
        (dims[2][0] for dims in model_dims.values() if dims[2][0] is not None and dims[2][0] == dims[2][0]), key=abs)

    for model_id in sorted_model_ids:
        name_display = get_model_display_name(model_id)
        width_me, width_std = model_dims[model_id][0]
        height_me, height_std = model_dims[model_id][1]
        depth_me, depth_std = model_dims[model_id][2]

        width_str = format_me_std(width_me, width_std)
        height_str = format_me_std(height_me, height_std)
        depth_str = format_me_std(depth_me, depth_std)

        if width_me == closest_width_me: width_str = f"\\textbf{{{width_str}}}"
        if height_me == closest_height_me: height_str = f"\\textbf{{{height_str}}}"
        if depth_me == closest_depth_me: depth_str = f"\\textbf{{{depth_str}}}"

        table.append(f"{name_display:<35} & {width_str} & {height_str} & {depth_str} \\\\")

    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\label{tab:dim_me}")
    table.append("\\end{center}")
    table.append("\\end{table}")

    return '\n'.join(table)


def generate_grasp_me_table(data: dict) -> str:
    """Generate the grasp parameters ME (bias) table with STD."""
    rot_me_vals = data['numeric_metrics']['Parsed_Hand_Rotation_Deg']['ME']
    rot_std_vals = data['numeric_metrics']['Parsed_Hand_Rotation_Deg']['ME_STD']
    ap_me_vals = data['numeric_metrics']['Parsed_Hand_Aperture_mm']['ME']
    ap_std_vals = data['numeric_metrics']['Parsed_Hand_Aperture_mm']['ME_STD']
    fing_me_vals = data['numeric_metrics']['Parsed_Num_Fingers']['ME']
    fing_std_vals = data['numeric_metrics']['Parsed_Num_Fingers']['ME_STD']

    all_models_ids = data['dataset_stats']['models']
    model_metrics_data = {}

    for model_id in all_models_ids:
        r_me, r_std = rot_me_vals.get(model_id), rot_std_vals.get(model_id)
        a_me, a_std = ap_me_vals.get(model_id), ap_std_vals.get(model_id)
        f_me, f_std = fing_me_vals.get(model_id), fing_std_vals.get(model_id)
        if all(v is not None and v == v for v in [r_me, r_std, a_me, a_std, f_me, f_std]):
            model_metrics_data[model_id] = {
                'rot': (r_me, r_std), 'ap': (a_me, a_std), 'fing': (f_me, f_std)
            }

    avg_abs_me_for_sort = {m: (abs(vals['rot'][0]) + abs(vals['ap'][0]) + abs(vals['fing'][0])) / 3.0
                           for m, vals in model_metrics_data.items()}
    sorted_model_ids = sort_models_by_metric(list(model_metrics_data.keys()), avg_abs_me_for_sort, metric_key=None)

    table = []
    table.append("\\begin{table}[t]")
    table.append("\\caption{ME ± STD for grasp parameters; ME closest to 0 highlighted in \\textbf{bold}.}")
    table.append("\\begin{center}")
    table.append("\\begin{tabular}{|l|c|c|c|}")
    table.append("\\hline")
    table.append("\\textbf{Model} & \\textbf{Rotation (°)} & \\textbf{Aperture (mm)} & \\textbf{Fingers} \\\\")
    table.append("\\hline")

    closest_rot_me = min((v['rot'][0] for v in model_metrics_data.values()), key=abs)
    closest_ap_me = min((v['ap'][0] for v in model_metrics_data.values()), key=abs)
    closest_fing_me = min((v['fing'][0] for v in model_metrics_data.values()), key=abs)

    for model_id in sorted_model_ids:
        name_display = get_model_display_name(model_id)
        metrics = model_metrics_data[model_id]

        rot_str = format_me_std(metrics['rot'][0], metrics['rot'][1])
        ap_str = format_me_std(metrics['ap'][0], metrics['ap'][1])
        fing_str = format_me_std(metrics['fing'][0], metrics['fing'][1])

        if metrics['rot'][0] == closest_rot_me: rot_str = f"\\textbf{{{rot_str}}}"
        if metrics['ap'][0] == closest_ap_me: ap_str = f"\\textbf{{{ap_str}}}"
        if metrics['fing'][0] == closest_fing_me: fing_str = f"\\textbf{{{fing_str}}}"

        table.append(f"{name_display:<35} & {rot_str} & {ap_str} & {fing_str} \\\\")

    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\label{tab:grasp_me}")
    table.append("\\end{center}")
    table.append("\\end{table}")
    return '\n'.join(table)


def generate_latency_cost_table(data: dict) -> str:
    """Generate the latency and cost table with STD for latency."""
    latency_cost_metrics = data['latency_cost_metrics']

    # Sort by average latency
    sorted_model_ids = sort_models_by_metric(
        list(latency_cost_metrics.keys()),
        latency_cost_metrics,
        metric_key='avg_latency'
    )

    table = []
    table.append("\\begin{table}[t]")
    table.append(
        "\\caption{Mean latency ± STD (s) and cost per query; best values highlighted in \\textbf{bold}.}")
    table.append("\\begin{center}")
    table.append("\\begin{tabular}{|l|c|c|}")
    table.append("\\hline")
    table.append("\\textbf{Model} & \\textbf{Latency (s)} & \\textbf{Cost (\\$)} \\\\")
    table.append("\\hline")

    # Find best values for bolding
    min_avg_latency = min(m['avg_latency'] for m in latency_cost_metrics.values() if
                          m['avg_latency'] is not None and m['avg_latency'] == m['avg_latency'])
    min_avg_cost = min(m['avg_cost'] for m in latency_cost_metrics.values() if
                       m['avg_cost'] is not None and m['avg_cost'] == m['avg_cost'])

    for model_id in sorted_model_ids:
        name_display = get_model_display_name(model_id)
        metrics = latency_cost_metrics[model_id]
        avg_lat = metrics['avg_latency']
        std_lat = metrics.get('std_latency')
        avg_c = metrics['avg_cost']

        lat_str = format_me_std(avg_lat, std_lat)
        cost_str = format_float(avg_c, 5)

        if avg_lat == min_avg_latency: lat_str = f"\\textbf{{{lat_str}}}"
        if avg_c == min_avg_cost: cost_str = f"\\textbf{{{cost_str}}}"

        table.append(f"{name_display:<35} & {lat_str} & {cost_str} \\\\")

    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\label{tab:latency}")
    table.append("\\end{center}")
    table.append("\\end{table}")
    return '\n'.join(table)


def main():
    """
    Generate all LaTeX tables.
    Tables are dynamically generated from metrics_results.json.
    """
    data = load_metrics('metrics_results.json')

    print("\\section{Results}")
    print("\\label{sec:results}")
    print()
    print("\\subsection{Object-Level Perception}")
    print()
    print("Categorical accuracies (Table~\\ref{tab:cat}) show that all models achieved high performance in object naming (82.4-97.1\\%), with Mistral Medium achieving the highest accuracy. For shape recognition, six models achieved perfect 100\\% accuracy, while the remaining two exceeded 97\\%. Object orientation was also recognized with high accuracy (85.3-100\\%), with Claude 3.5 and 3.7 Sonnet achieving perfect performance.")
    print()
    print("Grasp type classification showed more variation across models, ranging from 91.2\\% (Gemini Flash 17-04) down to 44.1\\% (Gemma 3). This parameter represents a binary choice between palmar and lateral grasps. The prompt specified that lateral grasps should pinch the object between the thumb and the radial side of the index finger, typically with the palm down (0°). Some models predicted palmar grasps with a 90° rotation for slender objects, such as markers, which is technically valid but less efficient than a lateral grasp.")
    print()
    print(generate_categorical_table(data))
    print()
    print("\\subsection{Object size and distance perception}")
    print()
    print(" Table~\\ref{tab:dim} shows the models' ability to estimate object dimensions from single images. Gemini 2.5 Flash 20-05 achieved the lowest MAE across all dimensions, with particularly strong performance in depth estimation (4.62 ± 4.06 mm).")
    print()
    print(generate_dimensions_mae_table(data))
    print()
    print("\\subsection{Estimation of hand rotation and aperture}")
    print()
    print("Table~\\ref{tab:grasp} presents the accuracy of grasp parameter estimation. Hand rotation showed the highest variability among all parameters, with MAEs ranging from 13.2° to 47.6°. Claude 3.5 Sonnet and Mistral Medium achieved the lowest MAE for finger count estimation (0.50), while GPT-4.1 showed the best hand aperture estimation (5.74 ± 6.14 mm).")
    print()
    print(generate_grasp_mae_table(data))
    print()
    print("\\subsection{Bias Analysis (Mean Error)}")
    print("Tables~\\ref{tab:dim_me} and \\ref{tab:grasp_me} show the ME analysis, revealing systematic biases in model predictions. A ME close to 0 indicates unbiased predictions, while positive and negative values indicate overestimation and underestimation, respectively.")
    print()
    print(generate_dimensions_me_table(data))
    print()
    print("For grasp parameters, GPT-4.1 and Claude 3.7 Sonnet showed minimal bias in aperture estimation (0.44 ± 8.39 mm and 0.15 ± 8.34 mm, respectively), which is favorable for practical use.")
    print()
    print(generate_grasp_me_table(data))
    print()
    print("\\subsection{Latency and Cost}")
    print()
    print("Table~\\ref{tab:latency} presents the practical deployment considerations. Gemini 2.5 Flash 20-05 achieved the fastest response time (2.60 ± 0.47 s), while Gemma 3 27B-IT offered the lowest cost per query (\\$0.00016). The Claude models showed the highest latencies (7-9 s) and costs (\\$0.012/query).")
    print()
    print(generate_latency_cost_table(data))


if __name__ == "__main__":
    main()