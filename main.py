import random
import sys
import pickle
import math



class Node:
    def __init__(self,left,right,index):
        self.left_subtree=left
        self.right_subtree=right
        self.col_index=index

class Leaf:
    def __init__(self, feature_matrix):
        self.output_prediction=max_output(feature_matrix)

def max_output(feature_matrix):
    en_total_count=0
    nl_total_count=0
    for line in feature_matrix:
        label=line[-1]
        if (label == 'en' or label == True):
            en_total_count += 1
        else:
            nl_total_count += 1
    return en_total_count,nl_total_count



def split_tree(feature_matrix,column_index):
    true_list = []
    false_list = []
    for line in feature_matrix:
        current_column_value=line[column_index]
        if current_column_value is True:
            true_list.append(line)
        else:
            false_list.append(line)
    return true_list,false_list




def calculate_entropy(feature_matrix,index):
    total_count = len(feature_matrix)
    entropy=0
    en_total_count = 0
    nl_total_count = 0
    for line in feature_matrix:
        label = line[index]
        if (label == 'en' or label == True):
            en_total_count += 1
        else:
            nl_total_count += 1
    if (total_count != 0):
        en_count_prob = en_total_count / total_count
        nl_count_prob = nl_total_count / total_count
        if en_count_prob != 0:
            entropy += en_count_prob * math.log(en_count_prob, 2)

        if nl_count_prob != 0:
            entropy += nl_count_prob * math.log(nl_count_prob, 2)
    else:
        entropy = 0
    return -(entropy)

def information_gain(feature_matrix,column_index):
    decision_entropy=calculate_entropy(feature_matrix,-1)
    true_list,false_list=split_tree(feature_matrix,column_index)
    prob_true_count=len(true_list)/(len(feature_matrix))
    prob_false_count=len(false_list)/(len(feature_matrix))
    info_gain=decision_entropy-((prob_true_count*calculate_entropy(true_list,-1))+(prob_false_count*calculate_entropy(false_list,-1)))
    return info_gain

def find_next_levelcolumn(feature_matrix):
    total_column=len(feature_matrix[0])
    info_gain=-(sys.maxsize)
    index=0
    for i in range(0,total_column):
        current_gain=information_gain(feature_matrix,i)
        if(current_gain>info_gain):
            info_gain=current_gain
            index=i
    return info_gain,index



def build_decision_tree(feature_matrix):
    find_maxGain,maxGain_column=find_next_levelcolumn(feature_matrix)
    if find_maxGain==0:
        return Leaf(feature_matrix)
    left, right = split_tree(feature_matrix, maxGain_column)
    left_subtree = build_decision_tree(left)
    right_subtree = build_decision_tree(right)
    return Node(left_subtree, right_subtree, maxGain_column)




def create_featureMatrix(line_list,label):
    line_wordSplit_list=[]
    dutch_personal_pronoun=["ik","jij","je","u","hij","-ie","zij","ze","het","wij","jullie","ze","mij","jou","je"\
                             "u","hem","haar","het","ons","jullie","hen","mijn","mijne","jouw","jouwe","uw"\
                            "uwe","zijn","haar","ons","onze","hun"\
                            ]
    english_personal_pronoun=["i","you","he","she","it","we","they","me","you","him","her","it",\
                              "us","them","your","his","our","their"]
    feature_matrix=[]
    for line in line_list:
        line_wordSplit_list.append(line.split())
    count=0
    for current_line in line_wordSplit_list:
        current_line_feature=[]
        q_found=False
        # Assigning True to English and False to Dutch
        if("the" in current_line):
             current_line_feature.append(True)
        else:
            current_line_feature.append(False)
        if("Een" in current_line):
            current_line_feature.append(False)
        else:
            current_line_feature.append(True)
        for word in current_line:
            if "q" in word:
                current_line_feature.append(True)
                q_found=True
                break
        if(not q_found):
            current_line_feature.append(False)
        if "you" in current_line:
            current_line_feature.append(True)
        else:
            current_line_feature.append(False)
        if "and" in current_line or "but" in current_line:
            current_line_feature.append(True)
        else:
            current_line_feature.append(False)
        for word in current_line:
            if word in dutch_personal_pronoun:
                current_line_feature.append(False)
                break
            else:
                current_line_feature.append(True)
                break
        if len(current_line) > 5:
            current_line_feature.append(True)
        else:
            current_line_feature.append(False)
        if "on" in current_line or "up" in current_line or "down" in current_line:
            current_line_feature.append(True)
        else:
            current_line_feature.append(False)
        if(label!=[]):
            current_line_feature.append(label[count])
        count += 1
        feature_matrix.append(current_line_feature)
    return feature_matrix




def dt_classification(input, node):
    if isinstance(node, Leaf):
        if (node.output_prediction[0] > node.output_prediction[1]):
            return 'en'
        else:
            return 'nl'
    elif input[node.col_index] == True:
        return dt_classification(input, node.left_subtree)
    else:
        return dt_classification(input, node.right_subtree)


def ada_boost(feature_matrix):
        w = []
        z = []
        h = []
        output = []
        count = 0
        temp = []
        newlyCreatedMatrix = []
        feature_matrix_copy = feature_matrix
        for elem in feature_matrix_copy:
            if (elem[-1] == 'en'):
                output.append(True)
            else:
                output.append(False)
        for k in range(len(feature_matrix[0]) - 1):
            sample_weight_matrix=[]
            for i in range(len(feature_matrix)):
                w.append(1 / len(feature_matrix))
            sample_weight_matrix.append(w[0])
            for i in (range(1,len(w))):
                sample_weight_matrix.append(w[i]+w[i-1])
            for _ in range(0,len(sample_weight_matrix)):
                random_weight=random.uniform(0,1)
                count=0
                while(sample_weight_matrix[count]<random_weight):
                    count+=1
                newlyCreatedMatrix.append(feature_matrix_copy[count])
            max_gain, index = find_next_levelcolumn(newlyCreatedMatrix)
            h.append(index)
            error = 0
            for j in range(len(newlyCreatedMatrix)):
                if newlyCreatedMatrix[j][k] != output[j]:
                    error += w[j]
            for j in range(len(newlyCreatedMatrix)):
                if newlyCreatedMatrix[j][k] == output[j]:
                    w[j] = w[j] * (error / (1 - error))
            total_weight = 0
            w_array = []
            for weights in w:
                total_weight += weights
            for weight in w:
                w_array.append(weight/total_weight)  #normalization
            if error == 1:
                z.append(0)
            elif error == 0:
                z.append(float('inf'))
            else:
                z.append(math.log((1 - error) / error))
        hypothesis = [h, z]
        return hypothesis





def main(input_data):
    line_list=[]
    new_line_list=[]
    label=[]
    toDo_train_or_test=input_data[0]
    file_name = input_data[1]
    hypothesisOut=input_data[2]
    learningType=input_data[3]
    if(toDo_train_or_test=="train"):
        with open(str(file_name), encoding="utf8") as f:
            for line in f:
                line = line.rstrip('\n')
                splited_list = line.split("|")
                label.append(splited_list[0].rstrip('\n'))
                line_list.extend(splited_list[1:])
        feature_matrix = create_featureMatrix(line_list, label)
        if learningType == "dt":
            hypothesis = build_decision_tree(feature_matrix)
        elif learningType == "ada":
            hypothesis = ada_boost(feature_matrix)
        print(hypothesis)
        pickle.dump(hypothesis, open(hypothesisOut, 'wb'))
        print("Training done")
    elif(toDo_train_or_test=="predict"):
        with open(str(file_name), encoding="utf8") as f:
            for line in f:
                line = line.rstrip('\n')
                line_list.append(line)
        tree=pickle.load(open(hypothesisOut, 'rb'))
        test_feature_matrix=create_featureMatrix(line_list,label)
        for input in test_feature_matrix:
            print(dt_classification(input,tree))
        print("Testing done")






if __name__ == '__main__':
    input_data=sys.argv
    main(input_data[1:])
