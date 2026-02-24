#!/usr/bin/env python3
"""
Example usage of the robot randomizer with different presets.
"""

from randomize_robot import RobotRandomizer
import os


def create_slight_variant(input_xml, output_xml, seed=42):
    """Create a slightly different robot (small variations)."""
    print("Creating slight variant...")
    randomizer = RobotRandomizer(input_xml, seed=seed)
    randomizer.randomize_all(
        mass_range=(0.8, 1.2),
        length_range=(0.9, 1.1),
        gear_range=(0.8, 1.2),
        damping_range=(0.8, 1.5),
        friction_range=(0.08, 0.15),
        surface_friction_range=(0.8, 1.2),
        removal_prob=0.0,
        add_position_noise=True
    )
    randomizer.save(output_xml)


def create_heavy_variant(input_xml, output_xml, seed=42):
    """Create a heavier, stronger robot."""
    print("Creating heavy variant...")
    randomizer = RobotRandomizer(input_xml, seed=seed)
    randomizer.randomize_all(
        mass_range=(1.5, 2.5),  # Heavier
        length_range=(1.1, 1.4),  # Longer limbs
        gear_range=(1.2, 2.0),  # Stronger motors
        damping_range=(1.5, 3.0),
        friction_range=(0.1, 0.3),
        surface_friction_range=(0.8, 1.2),
        removal_prob=0.0,
        add_position_noise=True
    )
    randomizer.save(output_xml)


def create_light_variant(input_xml, output_xml, seed=42):
    """Create a lighter, more agile robot."""
    print("Creating light variant...")
    randomizer = RobotRandomizer(input_xml, seed=seed)
    randomizer.randomize_all(
        mass_range=(0.4, 0.8),  # Lighter
        length_range=(0.7, 0.9),  # Shorter limbs
        gear_range=(0.4, 0.8),  # Weaker motors
        damping_range=(0.3, 1.0),
        friction_range=(0.03, 0.1),
        surface_friction_range=(0.6, 1.0),
        removal_prob=0.0,
        add_position_noise=True
    )
    randomizer.save(output_xml)


def create_damaged_variant(input_xml, output_xml, seed=42):
    """Create a robot with some missing/damaged components."""
    print("Creating damaged variant...")
    randomizer = RobotRandomizer(input_xml, seed=seed)
    randomizer.randomize_all(
        mass_range=(0.6, 1.4),
        length_range=(0.8, 1.2),
        gear_range=(0.5, 1.5),
        damping_range=(0.5, 2.5),
        friction_range=(0.05, 0.25),
        surface_friction_range=(0.5, 1.3),
        removal_prob=0.15,  # 15% chance to remove non-critical links
        add_position_noise=True
    )
    randomizer.save(output_xml)


def create_extreme_variant(input_xml, output_xml, seed=42):
    """Create a very different robot."""
    print("Creating extreme variant...")
    randomizer = RobotRandomizer(input_xml, seed=seed)
    randomizer.randomize_all(
        mass_range=(0.3, 3.0),  # Very wide range
        length_range=(0.5, 1.5),  # Very wide range
        gear_range=(0.3, 3.0),  # Very wide range
        damping_range=(0.3, 5.0),
        friction_range=(0.02, 0.5),
        surface_friction_range=(0.3, 2.0),
        removal_prob=0.0,
        add_position_noise=True
    )
    randomizer.save(output_xml)


if __name__ == '__main__':
    input_xml = '/home/sviswasam/dr/ModuMorph/modular/unitree_g1/xml/g1_12dof.xml'
    
    # Check if input file exists
    if not os.path.exists(input_xml):
        print(f"Error: {input_xml} not found!")
        print("Please save your MuJoCo XML as 'g1_12dof.xml' in the current directory.")
        exit(1)
    
    # Create different variants
    print("\n" + "="*60)
    print("Generating Robot Variants")
    print("="*60 + "\n")
    
    variants = [
        (create_slight_variant, 'robot_slight_variant.xml', 42),
        (create_heavy_variant, 'robot_heavy_variant.xml', 123),
        (create_light_variant, 'robot_light_variant.xml', 456),
        (create_damaged_variant, 'robot_damaged_variant.xml', 789),
        (create_extreme_variant, 'robot_extreme_variant.xml', 999),
    ]
    
    for create_func, output_file, seed in variants:
        print(f"\n{'='*60}")
        create_func(input_xml, output_file, seed)
    
    print(f"\n{'='*60}")
    print("All variants created successfully!")
    print(f"{'='*60}\n")
    
    print("Generated files:")
    for _, output_file, _ in variants:
        print(f"  - {output_file}")