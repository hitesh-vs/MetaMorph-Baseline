import xml.etree.ElementTree as ET
import numpy as np
import os
import json
import argparse


def scale_vector(vec_string, scale):
    vec = np.array([float(x) for x in vec_string.split()])
    vec = vec * scale
    return " ".join(map(str, vec))


def perturb_range(range_string, factor):
    low, high = map(float, range_string.split())
    center = (low + high) / 2
    width = (high - low) / 2
    width *= factor
    return f"{center - width} {center + width}"


def main(args):

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.num_variants):

        tree = ET.parse(args.base_xml)
        root = tree.getroot()

        # Sample random scalars
        mass_scale = np.random.uniform(0.8, 1.2)
        leg_length_scale = np.random.uniform(0.9, 1.1)
        joint_range_scale = np.random.uniform(0.95, 1.05)
        density_scale = np.random.uniform(0.9, 1.1)

        changes = {
            "mass_scale": float(mass_scale),
            "leg_length_scale": float(leg_length_scale),
            "joint_range_scale": float(joint_range_scale),
            "density_scale": float(density_scale),
        }

        # ----------------------
        # Mass + inertia scaling
        # ----------------------
        for inertial in root.iter("inertial"):
            if "mass" in inertial.attrib:
                inertial.attrib["mass"] = str(
                    float(inertial.attrib["mass"]) * mass_scale
                )

            if "diaginertia" in inertial.attrib:
                inertial.attrib["diaginertia"] = scale_vector(
                    inertial.attrib["diaginertia"], mass_scale
                )

        # ----------------------
        # Leg length scaling
        # ----------------------
        leg_links = {
            "left_hip_pitch_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "right_hip_pitch_link",
            "right_knee_link",
            "right_ankle_pitch_link",
        }

        for body in root.iter("body"):
            if body.attrib.get("name") in leg_links and "pos" in body.attrib:
                body.attrib["pos"] = scale_vector(
                    body.attrib["pos"], leg_length_scale
                )

        # ----------------------
        # Joint range perturbation
        # ----------------------
        for joint in root.iter("joint"):
            if "range" in joint.attrib:
                joint.attrib["range"] = perturb_range(
                    joint.attrib["range"], joint_range_scale
                )

        # ----------------------
        # Geom density scaling
        # ----------------------
        for geom in root.iter("geom"):
            if "density" in geom.attrib:
                geom.attrib["density"] = str(
                    float(geom.attrib["density"]) * density_scale
                )

        # ----------------------
        # Save files
        # ----------------------
        xml_name = f"robot_variant_{i}.xml"
        meta_name = f"robot_variant_{i}_changes.json"

        tree.write(os.path.join(args.out_dir, xml_name))
        with open(os.path.join(args.out_dir, meta_name), "w") as f:
            json.dump(changes, f, indent=4)

        print(f"[✓] Generated {xml_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate MuJoCo robot morphology variants"
    )

    parser.add_argument(
        "--base_xml",
        type=str,
        required=True,
        help="Path to base MuJoCo XML file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="generated_robots",
        help="Output directory",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=10,
        help="Number of variants to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional)",
    )

    args = parser.parse_args()
    main(args)