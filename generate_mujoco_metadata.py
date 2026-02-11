#!/usr/bin/env python3
"""
MuJoCo XML Metadata Generator

This script processes MuJoCo XML files and generates corresponding metadata JSON files
containing information about symmetric limbs, degrees of freedom, and number of limbs.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
import argparse


def count_actuators(root: ET.Element) -> int:
    """Count the number of actuators (degrees of freedom) in the model."""
    actuator_elem = root.find('actuator')
    if actuator_elem is None:
        return 0
    return len(actuator_elem.findall('motor'))


def count_limbs(root: ET.Element) -> int:
    """Count the number of body elements (limbs) in the model."""
    worldbody = root.find('worldbody')
    if worldbody is None:
        return 0
    
    # Count all body elements recursively
    def count_bodies(element):
        count = 0
        for body in element.findall('.//body'):
            count += 1
        return count
    
    return count_bodies(worldbody)


def detect_symmetric_limbs(root: ET.Element) -> List[str]:
    """
    Detect symmetric limbs in the model.
    This is a heuristic approach looking for paired body names.
    """
    worldbody = root.find('worldbody')
    if worldbody is None:
        return []
    
    # Get all body names
    body_names = []
    for body in worldbody.findall('.//body'):
        name = body.get('name')
        if name:
            body_names.append(name)
    
    # Look for common symmetric patterns
    symmetric_pairs = []
    left_right_pairs = {}
    
    for name in body_names:
        # Check for left/right patterns
        if 'left' in name.lower():
            right_name = name.lower().replace('left', 'right')
            if right_name in [n.lower() for n in body_names]:
                pair_name = name.lower().replace('left', '').strip('_- ')
                if pair_name not in left_right_pairs:
                    left_right_pairs[pair_name] = True
                    symmetric_pairs.append(name)
        elif 'right' in name.lower():
            left_name = name.lower().replace('right', 'left')
            if left_name in [n.lower() for n in body_names]:
                pair_name = name.lower().replace('right', '').strip('_- ')
                if pair_name not in left_right_pairs:
                    left_right_pairs[pair_name] = True
                    symmetric_pairs.append(name)
        # Check for _l/_r patterns
        elif name.endswith('_l') or name.endswith('_L'):
            right_name = name[:-2] + '_r'
            if right_name in body_names or right_name.upper() in body_names:
                if name[:-2] not in left_right_pairs:
                    left_right_pairs[name[:-2]] = True
                    symmetric_pairs.append(name)
        elif name.endswith('_r') or name.endswith('_R'):
            left_name = name[:-2] + '_l'
            if left_name in body_names or left_name.upper() in body_names:
                if name[:-2] not in left_right_pairs:
                    left_right_pairs[name[:-2]] = True
                    symmetric_pairs.append(name)
    
    return sorted(symmetric_pairs)


def generate_metadata(xml_path: str) -> Dict:
    """Generate metadata dictionary from a MuJoCo XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        metadata = {
            "symmetric_limbs": detect_symmetric_limbs(root),
            "dof": count_actuators(root),
            "num_limbs": count_limbs(root)
        }
        
        return metadata
    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return None


def process_xml_files(input_folder: str, output_folder: str = None, overwrite: bool = False):
    """
    Process all XML files in the input folder and generate corresponding JSON metadata files.
    
    Args:
        input_folder: Path to folder containing XML files
        output_folder: Path to folder where JSON files will be saved (default: same as input_folder)
        overwrite: Whether to overwrite existing JSON files
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        return
    
    # Use input folder as output folder if not specified
    if output_folder is None:
        output_path = input_path
    else:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all XML files
    xml_files = list(input_path.glob('*.xml'))
    
    if not xml_files:
        print(f"No XML files found in '{input_folder}'.")
        return
    
    print(f"Found {len(xml_files)} XML file(s) in '{input_folder}'.")
    print(f"Output folder: '{output_path}'")
    print("-" * 60)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for xml_file in xml_files:
        # Generate output JSON filename
        json_filename = xml_file.stem + '.json'
        json_path = output_path / json_filename
        
        # Check if JSON already exists
        if json_path.exists() and not overwrite:
            print(f"Skipping '{xml_file.name}' - JSON already exists")
            skipped += 1
            continue
        
        print(f"Processing '{xml_file.name}'...", end=' ')
        
        # Generate metadata
        metadata = generate_metadata(str(xml_file))
        
        if metadata is None:
            print("ERROR")
            errors += 1
            continue
        
        # Save JSON file
        try:
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"OK -> '{json_filename}'")
            processed += 1
        except Exception as e:
            print(f"ERROR writing JSON: {e}")
            errors += 1
    
    print("-" * 60)
    print(f"Summary: {processed} processed, {skipped} skipped, {errors} errors")


def main():
    parser = argparse.ArgumentParser(
        description='Generate metadata JSON files from MuJoCo XML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process XML files in current directory
  python generate_mujoco_metadata.py .
  
  # Process XML files and save JSON to different folder
  python generate_mujoco_metadata.py ./xml_files ./metadata
  
  # Overwrite existing JSON files
  python generate_mujoco_metadata.py ./xml_files --overwrite
        """
    )
    
    parser.add_argument(
        'input_folder',
        help='Path to folder containing MuJoCo XML files'
    )
    
    parser.add_argument(
        'output_folder',
        nargs='?',
        default=None,
        help='Path to folder where JSON files will be saved (default: same as input_folder)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing JSON files'
    )
    
    args = parser.parse_args()
    
    process_xml_files(args.input_folder, args.output_folder, args.overwrite)


if __name__ == '__main__':
    main()