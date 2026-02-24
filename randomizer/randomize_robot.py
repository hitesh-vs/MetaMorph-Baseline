#!/usr/bin/env python3
"""
Randomize MuJoCo robot physical parameters while maintaining structural validity.
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse
from copy import deepcopy


class RobotRandomizer:
    def __init__(self, xml_path, seed=None):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        if seed is not None:
            np.random.seed(seed)
    
    def randomize_masses(self, scale_range=(0.5, 2.0)):
        """Randomize inertial masses by a scaling factor."""
        print("Randomizing masses...")
        for inertial in self.root.iter('inertial'):
            if 'mass' in inertial.attrib:
                original_mass = float(inertial.attrib['mass'])
                scale = np.random.uniform(*scale_range)
                new_mass = original_mass * scale
                inertial.attrib['mass'] = f"{new_mass:.6f}"
                
                # Also scale inertia proportionally
                if 'diaginertia' in inertial.attrib:
                    inertias = [float(x) for x in inertial.attrib['diaginertia'].split()]
                    # Inertia scales with mass and length^2, so we use scale^(4/3) as approximation
                    inertia_scale = scale ** (4/3)
                    new_inertias = [i * inertia_scale for i in inertias]
                    inertial.attrib['diaginertia'] = ' '.join([f"{i:.8f}" for i in new_inertias])
    
    def randomize_link_lengths(self, scale_range=(0.7, 1.3)):
        """Randomize link lengths by scaling geometry and positions."""
        print("Randomizing link lengths...")
        
        # Track bodies and their scale factors
        body_scales = {}
        
        for body in self.root.iter('body'):
            body_name = body.attrib.get('name', '')
            
            # Skip pelvis/root to maintain grounding
            if body_name == 'pelvis':
                body_scales[body_name] = 1.0
                continue
            
            scale = np.random.uniform(*scale_range)
            body_scales[body_name] = scale
            
            # Scale geom dimensions
            for geom in body.findall('geom'):
                if 'fromto' in geom.attrib:
                    # Scale capsule/cylinder lengths
                    fromto = [float(x) for x in geom.attrib['fromto'].split()]
                    fromto = [x * scale for x in fromto]
                    geom.attrib['fromto'] = ' '.join([f"{x:.6f}" for x in fromto])
                
                if 'size' in geom.attrib:
                    # Scale radii moderately
                    sizes = [float(x) for x in geom.attrib['size'].split()]
                    radius_scale = np.random.uniform(0.8, 1.2)
                    sizes = [x * radius_scale for x in sizes]
                    geom.attrib['size'] = ' '.join([f"{x:.6f}" for x in sizes])
                
                if 'pos' in geom.attrib:
                    # Scale positions
                    pos = [float(x) for x in geom.attrib['pos'].split()]
                    pos = [x * scale for x in pos]
                    geom.attrib['pos'] = ' '.join([f"{x:.6f}" for x in pos])
            
            # Scale child body positions to maintain connectivity
            for child_body in body.findall('body'):
                if 'pos' in child_body.attrib:
                    pos = [float(x) for x in child_body.attrib['pos'].split()]
                    pos = [x * scale for x in pos]
                    child_body.attrib['pos'] = ' '.join([f"{x:.6f}" for x in pos])
    
    def randomize_gears(self, scale_range=(0.5, 2.0)):
        """Randomize motor gear ratios."""
        print("Randomizing gear ratios...")
        for motor in self.root.iter('motor'):
            if 'gear' in motor.attrib:
                original_gear = float(motor.attrib['gear'])
                scale = np.random.uniform(*scale_range)
                new_gear = original_gear * scale
                motor.attrib['gear'] = f"{new_gear:.1f}"
    
    def randomize_joint_params(self, damping_range=(0.5, 3.0), friction_range=(0.05, 0.3)):
        """Randomize joint damping and friction."""
        print("Randomizing joint parameters...")
        for joint in self.root.iter('joint'):
            # Randomize damping
            if 'damping' in joint.attrib:
                damping = np.random.uniform(*damping_range)
                joint.attrib['damping'] = f"{damping:.2f}"
            
            # Randomize friction loss
            if 'frictionloss' in joint.attrib or True:
                friction = np.random.uniform(*friction_range)
                joint.attrib['frictionloss'] = f"{friction:.3f}"
            
            # Slightly randomize joint ranges (keep within reasonable bounds)
            if 'range' in joint.attrib:
                range_vals = [float(x) for x in joint.attrib['range'].split()]
                perturbation = np.random.uniform(-5, 5, size=2)
                new_range = [range_vals[0] + perturbation[0], range_vals[1] + perturbation[1]]
                # Ensure min < max
                if new_range[0] < new_range[1]:
                    joint.attrib['range'] = f"{new_range[0]:.1f} {new_range[1]:.1f}"
    
    def randomize_friction(self, scale_range=(0.5, 1.5)):
        """Randomize surface friction coefficients."""
        print("Randomizing friction coefficients...")
        for geom in self.root.iter('geom'):
            if 'friction' in geom.attrib:
                frictions = [float(x) for x in geom.attrib['friction'].split()]
                scales = [np.random.uniform(*scale_range) for _ in frictions]
                new_frictions = [f * s for f, s in zip(frictions, scales)]
                geom.attrib['friction'] = ' '.join([f"{x:.2f}" for x in new_frictions])
    
    def remove_random_links(self, removal_probability=0.1):
        """Randomly remove non-critical links/joints while maintaining connectivity."""
        print("Randomly removing links...")
        
        # Define critical bodies that should never be removed
        critical_bodies = {'pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link'}
        
        bodies_to_remove = []
        
        for body in self.root.iter('body'):
            body_name = body.attrib.get('name', '')
            
            # Skip critical bodies and root
            if body_name in critical_bodies or body_name == 'pelvis':
                continue
            
            # Randomly decide to remove
            if np.random.random() < removal_probability:
                # Check if body has children - if so, skip to maintain connectivity
                child_bodies = list(body.findall('body'))
                if len(child_bodies) == 0:  # Only remove leaf bodies
                    bodies_to_remove.append((body, body_name))
        
        # Remove selected bodies
        for parent in self.root.iter('body'):
            for body, body_name in bodies_to_remove:
                try:
                    parent.remove(body)
                    print(f"  Removed: {body_name}")
                    
                    # Also remove corresponding actuator
                    for motor in self.root.findall('.//actuator/motor'):
                        motor_name = motor.attrib.get('name', '')
                        if body_name in motor_name:
                            actuator = self.root.find('.//actuator')
                            if actuator is not None:
                                actuator.remove(motor)
                except ValueError:
                    pass  # Body not in this parent
    
    def add_noise_to_positions(self, noise_level=0.01):
        """Add small noise to body positions."""
        print("Adding positional noise...")
        for body in self.root.iter('body'):
            if 'pos' in body.attrib:
                pos = [float(x) for x in body.attrib['pos'].split()]
                noise = np.random.normal(0, noise_level, size=3)
                new_pos = [p + n for p, n in zip(pos, noise)]
                body.attrib['pos'] = ' '.join([f"{x:.6f}" for x in new_pos])
    
    def randomize_all(self, 
                     mass_range=(0.5, 2.0),
                     length_range=(0.7, 1.3),
                     gear_range=(0.5, 2.0),
                     damping_range=(0.5, 3.0),
                     friction_range=(0.05, 0.3),
                     surface_friction_range=(0.5, 1.5),
                     removal_prob=0.0,
                     add_position_noise=True):
        """Apply all randomizations."""
        print("\n=== Randomizing Robot Parameters ===\n")
        
        self.randomize_masses(mass_range)
        self.randomize_link_lengths(length_range)
        self.randomize_gears(gear_range)
        self.randomize_joint_params(damping_range, friction_range)
        self.randomize_friction(surface_friction_range)
        
        if removal_prob > 0:
            self.remove_random_links(removal_prob)
        
        if add_position_noise:
            self.add_noise_to_positions()
        
        print("\n=== Randomization Complete ===\n")
    
    def save(self, output_path):
        """Save the randomized XML."""
        # Format the XML nicely
        self._indent(self.root)
        self.tree.write(output_path, encoding='unicode', xml_declaration=True)
        print(f"Saved randomized model to: {output_path}")
    
    def _indent(self, elem, level=0):
        """Add indentation to XML for readability."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


def main():
    parser = argparse.ArgumentParser(description='Randomize MuJoCo robot physical parameters')
    parser.add_argument('input_xml', help='Input MuJoCo XML file')
    parser.add_argument('-o', '--output', default='randomized_robot.xml', 
                       help='Output XML file (default: randomized_robot.xml)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--mass-range', nargs=2, type=float, default=[0.5, 2.0],
                       help='Mass scaling range (default: 0.5 2.0)')
    parser.add_argument('--length-range', nargs=2, type=float, default=[0.7, 1.3],
                       help='Length scaling range (default: 0.7 1.3)')
    parser.add_argument('--gear-range', nargs=2, type=float, default=[0.5, 2.0],
                       help='Gear ratio scaling range (default: 0.5 2.0)')
    parser.add_argument('--remove-prob', type=float, default=0.0,
                       help='Probability of removing non-critical links (default: 0.0)')
    parser.add_argument('--no-noise', action='store_true',
                       help='Disable positional noise')
    
    args = parser.parse_args()
    
    # Create randomizer
    randomizer = RobotRandomizer(args.input_xml, seed=args.seed)
    
    # Apply randomizations
    randomizer.randomize_all(
        mass_range=tuple(args.mass_range),
        length_range=tuple(args.length_range),
        gear_range=tuple(args.gear_range),
        removal_prob=args.remove_prob,
        add_position_noise=not args.no_noise
    )
    
    # Save result
    randomizer.save(args.output)


if __name__ == '__main__':
    main()