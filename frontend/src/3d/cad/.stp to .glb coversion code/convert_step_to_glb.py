import FreeCAD as App
import ImportGui
import os
import json

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

STEP_PATH = r"C:\Users\newuser\OneDrive\Desktop\Rotax main\Rotax 914.stp"
OUTPUT_GLB = r"C:\Users\newuser\OneDrive\Desktop\Rotax main\rotax_914.glb"
OUTPUT_NAME_MAP = r"C:\Users\newuser\OneDrive\Desktop\Rotax main\PART_NAME_MAP.json"

def main():

    print("\n========================================")
    print(" Rotax 914 STEP -> GLB Converter")
    print("========================================")

    # -----------------------------------------------------
    # Check STEP file
    # -----------------------------------------------------

    if not os.path.exists(STEP_PATH):
        print("ERROR: STEP file not found:")
        print(STEP_PATH)
        return

    print("STEP file:")
    print(STEP_PATH)

    # -----------------------------------------------------
    # Create FreeCAD document
    # -----------------------------------------------------

    print("\nCreating FreeCAD document...")

    doc = App.newDocument("Rotax914")

    # -----------------------------------------------------
    # Import STEP
    # -----------------------------------------------------

    print("Importing STEP...")

    ImportGui.insert(
        STEP_PATH,
        doc.Name
    )

    doc.recompute()

    print(
        f"STEP import completed. Objects: {len(doc.Objects)}"
    )

    # -----------------------------------------------------
    # Collect objects
    # -----------------------------------------------------

    exported_objects = []
    name_map = {}

    print("\nFinding parts...")

    for obj in doc.Objects:

        if not hasattr(obj, "Shape"):
            continue

        if obj.Shape.isNull():
            continue

        if len(obj.Shape.Solids) == 0 and len(obj.Shape.Faces) == 0:
            continue

        # Original FreeCAD name
        original_name = obj.Name

        # Clean label
        clean_name = obj.Label.strip()

        clean_name = clean_name.replace(" ", "_")
        clean_name = clean_name.replace(".", "_")

        if not clean_name:
            clean_name = original_name

        # Set clean label
        obj.Label = clean_name

        name_map[original_name] = clean_name

        exported_objects.append(obj)

        print(
            f"  Found part: {clean_name}"
        )

    # -----------------------------------------------------
    # Check
    # -----------------------------------------------------

    if not exported_objects:

        print("\nERROR: No parts found.")

        return

    print(
        f"\nTotal parts found: {len(exported_objects)}"
    )

    # -----------------------------------------------------
    # Export GLB
    # -----------------------------------------------------

    print("\nExporting GLB...")
    print(OUTPUT_GLB)

    try:

        ImportGui.export(
            exported_objects,
            OUTPUT_GLB
        )

    except Exception as e:

        print("\nERROR during GLB export:")
        print(str(e))

        return

    # -----------------------------------------------------
    # Check GLB
    # -----------------------------------------------------

    if os.path.exists(OUTPUT_GLB):

        size_mb = (
            os.path.getsize(OUTPUT_GLB)
            / (1024 * 1024)
        )

        print("\n========================================")
        print(" GLB EXPORT SUCCESSFUL")
        print("========================================")

        print(
            f"GLB: {OUTPUT_GLB}"
        )

        print(
            f"Size: {size_mb:.2f} MB"
        )

    else:

        print("\nERROR: GLB file was NOT created.")

        return

    # -----------------------------------------------------
    # Save name map
    # -----------------------------------------------------

    with open(
        OUTPUT_NAME_MAP,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            name_map,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nName map: {OUTPUT_NAME_MAP}"
    )

    print(
        f"Total exported parts: {len(exported_objects)}"
    )

    print("\n========================================")
    print(" DONE")
    print("========================================")


# IMPORTANT:
# FreeCADCmd executes this script directly,
# so call main() explicitly.

main()