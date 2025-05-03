import os
import re
import csv


directory = "./"
header = "Outline,Conf,PupilDiameter,Eyelids,RectPoints_0_0,RectPoints_0_1,RectPoints_1_0,RectPoints_1_1,RectPoints_2_0,RectPoints_2_1,RectPoints_3_0,RectPoints_3_1,Size_0,Size_1,Major Axis,Minor Axis,Width,Height\n"

# Loop through all .txt files
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".txt"):
        txt_path = os.path.join(directory, filename)
        csv_path = os.path.join(directory, filename.replace(".txt", ".csv"))

        with open(txt_path, "r", encoding="utf-8") as txt_file:
            lines = txt_file.readlines()

        lines = lines[1:]

        lines = "".join(lines)

        lines = re.sub(r"\[|\]|\(|\)|,|True|False", "", lines)
        lines = re.sub(r"\n\t", "\n", lines)
        lines = re.sub(r"(?<!^)[ \t]+", ",", lines)

        # Write to .csv
        with open(csv_path, "w") as csv_file:
            csv_file.write(header)
            csv_file.write(lines)
        csv_file.close()

        # Delete the original .txt file
        # os.remove(txt_path)

print("Conversion and cleanup completed.")
