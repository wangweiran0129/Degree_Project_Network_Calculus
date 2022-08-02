# This code is written by Weiran Wang
# For any questions or problems, please contact the author of code at (weiran.wang@epfl.ch)

import csv, os, re
from collections import Counter

def potential_attack_target_pred1(pred_file):
    """
    For the first prediction classification:
    When the prediction values are close to 0.5
    Then it's the potential attack target
    :param pred_file: the file in the path of the prediction files
    :return: True if there exists a potential target in this file, otherwise false
    """
    fois_line = []
    fois = []
    PRED1 = []

    # Grab the topology id from the file name
    topology_id = int(re.findall(r'\d+', pred_file)[0])
    print("topology id : ", topology_id)

    # Navigate to the PRED1 chunk
    file = csv.reader(open(pred_file))
    for line, content in enumerate(file):
        if content[0] == "foi":
            fois_line.append(line+1)

    # Grab the flow of interest value
    file = csv.reader(open(pred_file))
    for line, content in enumerate(file):
        if line in fois_line:
            fois.append(int(content[0]))
            PRED1.append(float(content[3]))

    potential_attack_target = []
    
    for index in range(len(fois)):
        if abs(PRED1[index] - 0.5) < 0.1:
            potential_attack_target.append([topology_id, fois])

    return potential_attack_target


def potential_attack_target_pred2(pred_file):
    """
    For the second prediction classification
    We care more about the biggest value and some values close to the biggest value in a flow
    It is then the potential attack target
    :param pred_file: the file in the path of the prediction files
    """

    # Grab the topology id from the file name
    topology_id = int(re.findall(r'\d+', pred_file)[0])
    print("topology id : ", topology_id)

    flow_start_line = []
    flow_end_line = []

    # Get the line of foi in this topology
    fois_line = []
    fois = []
    file = csv.reader(open(pred_file))
    for line, content in enumerate(file):
        if content[0] == "foi":
            fois_line.append(line+1)

    # Get the foi value in this topology
    file = csv.reader(open(pred_file))
    for line, content in enumerate(file):
        if line in fois_line:
            fois.append(int(content[0]))

    # Navigate to the PRED2 chunk
    file = csv.reader(open(pred_file))
    file_total_length = 0
    for line, content in enumerate(file):
        file_total_length += 1
        if content[0] == "flow id":
            flow_start_line.append(line+1)
        if content[0] == "\n":
            flow_end_line.append(line-1)
    flow_end_line.append(file_total_length-1)

    # Grab the flow id of each flow of interest and PRED2
    flow_id = []
    PREDs = []
    file = csv.reader(open(pred_file))
    for index in range(len(flow_start_line)):
        flow_id.append([])
        PREDs.append([])
        file = csv.reader(open(pred_file))
        for line, content in enumerate(file):
            if line >= flow_start_line[index] and line <= flow_end_line[index]:
                flow_id[index].append(int(content[0]))
                PREDs[index].append(float(content[3]))

    potential_attack_target = []
    
    # Map the number of potential flow prolongations to the flow id
    for foi_index, foi in enumerate(fois):
        flow_counter = sorted(Counter(flow_id[foi_index]).items(), key=lambda d: d[0], reverse=False)
        counter = 0
        for flow_index in range(len(flow_counter)):
            flow_pred = []
            for pred_index in range(flow_counter[flow_index][1]):
                flow_pred.append(PREDs[foi_index][counter])
                counter = counter + 1
            flow_pred_max = max(flow_pred)
            # Keep a record on the topology id, flow of interest id, and the flow id
            flow_pred.remove(flow_pred_max)
            for pred_value in flow_pred:
                if abs(flow_pred_max - pred_value) <= 0.01 and flow_pred_max != pred_value:
                    potential_attack_target.append([topology_id, foi, flow_counter[flow_index][0]])
        print("potential attack target : ", potential_attack_target)
    return potential_attack_target


def main():

    main_path = "/Users/wangweiran/Desktop/MasterDegreeProject/Degree_Project_Network_Calculus/"

    # For pmoo
    pred_path = main_path + "Network_Information_and_Analysis/"

    pred_files = os.listdir(pred_path)
    if ".DS_Store" in pred_files:
        pred_files.remove(".DS_Store")
    pred_files.sort(key = lambda x: int(re.findall(r'\d+', x)[0]))

    potential_attack_target1_csv = main_path + "Network_Information_and_Analysis/potential_attack_target1.csv"
    potential_attack_target2_csv = main_path + "Network_Information_and_Analysis/potential_attack_target2.csv"
    count1 = 0
    count2 = 0

    for pred_file in pred_files:
        potential_attack_target1 = potential_attack_target_pred1(pred_path + pred_file)
        if len(potential_attack_target1) != 0:
            if count1 == 0 :
                with open(potential_attack_target1_csv, "w") as csvfile:
                    pred1_csv = csv.writer(csvfile)
                    pred1_csv.writerow(["topology id", "foi id", "flow id"])
                    for i in range(len(potential_attack_target1)):
                        pred1_csv.writerow([potential_attack_target1[i][0], potential_attack_target1[i][1]])
            else:
                with open(potential_attack_target1_csv, "+a") as csvfile:
                    pred1_csv = csv.writer(csvfile)
                    pred1_csv.writerow(["topology id", "foi id", "flow id"])
                    for i in range(len(potential_attack_target1)):
                        pred1_csv.writerow([potential_attack_target1[i][0], potential_attack_target1[i][1]])
            count1 = count1 + 1
        
        potential_attack_target2 = potential_attack_target_pred2(pred_path + pred_file)
        if len(potential_attack_target2) != 0:
            if count2 == 0:
                with open(potential_attack_target2_csv, "w") as csvfile:
                    pred2_csv = csv.writer(csvfile)
                    pred2_csv.writerow(["topology id", "foi id", "flow id"])
                    for i in range(len(potential_attack_target2)):
                        pred2_csv.writerow([potential_attack_target2[i][0], potential_attack_target2[i][1], potential_attack_target2[i][2]])
            else:
                with open(potential_attack_target2_csv, "+a") as csvfile:
                    pred2_csv = csv.writer(csvfile)
                    pred2_csv.writerow(["topology id", "foi id", "flow id"])
                    for i in range(len(potential_attack_target2)):
                        pred2_csv.writerow([potential_attack_target2[i][0], potential_attack_target2[i][1], potential_attack_target2[i][2]])
            count2 = count2 + 1


if __name__ == "__main__":
    main()