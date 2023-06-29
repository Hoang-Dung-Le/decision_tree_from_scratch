from fastapi import FastAPI, UploadFile, Form, File
from pydantic import BaseModel
import pandas as pd
from decision_tree_id3 import DecisionTreeID3
import json
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List

import numpy as np
from collections import Counter

class Node:
    def __init__(self, attribute=None, threshold=None, label=None):
        self.attribute = attribute  # Thuộc tính của nút
        self.threshold = threshold  # Ngưỡng (nếu thuộc tính liên tục)
        self.label = label  # Nhãn của nút (nếu là lá)

        self.children = {}  # Các nút con

    def add_child(self, value, node):
        self.children[value] = node

    def predict(self, sample):
        if self.label is not None:
            return self.label

        attribute_value = sample[self.attribute]

        if self.attribute in continuous_attributes:
            if attribute_value <= self.threshold:
                child_node = self.children["<="]
            else:
                child_node = self.children[">"]
        else:
            child_node = self.children[attribute_value]

        return child_node.predict(sample)


def entropy(y):
    counter = Counter(y)
    probabilities = [count / len(y) for count in counter.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy


def information_gain(X, y, attribute, threshold=None):
    if threshold is None:
        values = set(X[:, attribute])
    else:
        values = {"<=", ">"}

    subset_entropy = 0
    for value in values:
        if threshold is None:
            subset_indices = X[:, attribute] == value
        else:
            if value == "<=":
                subset_indices = X[:, attribute] <= threshold
            else:
                subset_indices = X[:, attribute] > threshold

        subset_y = y[subset_indices]
        subset_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    return entropy(y) - subset_entropy


def choose_best_attribute(X, y):
    best_gain = 0
    best_attribute = None
    best_threshold = None

    for attribute in range(X.shape[1]):
        if attribute in continuous_attributes:
            values = set(X[:, attribute])
            for value in values:
                gain = information_gain(X, y, attribute, threshold=value)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attribute
                    best_threshold = value
        else:
            gain = information_gain(X, y, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute

    return best_attribute, best_threshold


def create_decision_tree(X, y):
    if len(set(y)) == 1:
        # Nếu tất cả các nhãn giống nhau, tạo nút lá
        return Node(label=y[0])

    if X.shape[1] == 0:
        # Nếu không còn thuộc tính, tạo nút lá với nhãn phổ biến nhất
        most_common_label = Counter(y).most_common(1)[0][0]
        return Node(label=most_common_label)

    best_attribute, best_threshold = choose_best_attribute(X, y)

    if best_attribute is None:
        # Nếu không tìm thấy thuộc tính phù hợp, tạo nút lá với nhãn phổ biến nhất
        most_common_label = Counter(y).most_common(1)[0][0]
        return Node(label=most_common_label)

    node = Node(attribute=best_attribute, threshold=best_threshold)

    if best_attribute in continuous_attributes:
        for value in {"<=", ">"}:
            if value == "<=":
                subset_indices = X[:, best_attribute] <= best_threshold
            else:
                subset_indices = X[:, best_attribute] > best_threshold

            subset_X = X[subset_indices]
            subset_y = y[subset_indices]

            if len(subset_X) == 0:
                # Nếu không còn dữ liệu, tạo nút lá với nhãn phổ biến nhất
                most_common_label = Counter(y).most_common(1)[0][0]
                node.add_child(value, Node(label=most_common_label))
            else:
                node.add_child(value, create_decision_tree(subset_X, subset_y))
    else:
        values = set(X[:, best_attribute])
        for value in values:
            subset_indices = X[:, best_attribute] == value
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]

            if len(subset_X) == 0:
                # Nếu không còn dữ liệu, tạo nút lá với nhãn phổ biến nhất
                most_common_label = Counter(y).most_common(1)[0][0]
                node.add_child(value, Node(label=most_common_label))
            else:
                node.add_child(value, create_decision_tree(subset_X, subset_y))

    return node

def decision_tree_to_dict(node, attribute_mapping, parent=None, th=None):
    node_dict = {}

    # Thêm thuộc tính attribute và tên thuộc tính (attribute_name)
    if node.attribute is not None:
        attribute_name = attribute_mapping.get(node.attribute)
        node_dict["attribute"] = attribute_name
    else:
        node_dict["attribute"] = None

    # Thêm thuộc tính value
    if parent is not None:
        for value, child_node in parent.children.items():
            if child_node == node:
                node_dict["value"] = value
                break
    else:
        node_dict["value"] = None

    # Thêm thuộc tính threshold (chỉ cho thuộc tính liên tục)
    # if node.attribute in continuous_attributes:
    #     node_dict["threshold"] = node.threshold
    # else:
    #     node_dict["threshold"] = None

    node_dict["threshold"] = th

    # Thêm thuộc tính parent (tên của nút cha)
    if parent is not None:
        node_dict["parent"] = parent.attribute
    else:
        node_dict["parent"] = None

    # Thêm thuộc tính label
    node_dict["label"] = node.label

    # Tạo danh sách các nút con
    children = []
    for child_node in node.children.values():
        if node.attribute in continuous_attributes:
            child_dict = decision_tree_to_dict(child_node, attribute_mapping, parent=node, th=node.threshold)
            children.append(child_dict)
        else: 
            child_dict = decision_tree_to_dict(child_node, attribute_mapping, parent=node, th=None)
            children.append(child_dict)

    node_dict["children"] = children

    return node_dict


def predict_samples(X, tree):
    return [tree.predict(sample) for sample in X]

class NodeJson:
    def __init__(self, split_attribute, label, order, parent):
        self.split_attribute=split_attribute
        self.label=label
        self.order=order
        self.parent=parent
    def print_info(self):
        print(self.order)
        print(self.label)
        print(self.parent)
        print(self.split_attribute)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể chỉ định các nguồn cụ thể nếu cần
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_arr(arr, json_tree):
    stack = [(json_tree, None)]  # Sử dụng stack thay vì đệ quy
    while stack:
        node, parent = stack.pop()
        order = node['value']
        if node['threshold'] != None:
            order = str(order) + str(node['threshold'])
        arr.append(NodeJson(node['attribute'], node['label'], order, parent=parent))
        for child in reversed(node['children']):  # Đảo ngược thứ tự để duyệt theo thứ tự đúng
            stack.append((child, node['attribute']))


@app.post("/decision-tree-c45")
async def decision_tree_c45(file: Annotated[UploadFile, Form()], conti_attribute: Annotated[str, Form()]):
    if file is None:
        return {"message": "No file received"}
    # Read the CSV file into a DataFrame
    # print(conti_attribute)
    try:
        global continuous_attributes
        if conti_attribute == 'empty':
            continuous_attributes = set()
        else:
            continuous_attributes = [int(num) for num in conti_attribute.split(",")]
        # print(continuous_attributes)
        data = pd.read_csv(file.file)
        # print(data.head())
        attribute_name = data.columns.values.tolist()
        attribute_name_dict = {}
        for i in range(len(attribute_name) - 1):
            attribute_name_dict.update({i: attribute_name[i]})
        # print(attribute_name_dict)
        # data = data.iloc[1:]

        # Chuyển đổi dữ liệu thành mảng numpy
        data_np = np.array(data)

        # Tách thuộc tính và nhãn
        X = data_np[:, :-1]  # Thuộc tính
        y = data_np[:, -1]  # Nhãn

        # Tạo cây quyết định
        decision_tree = create_decision_tree(X, y)
        decision_tree_dict = decision_tree_to_dict(decision_tree, attribute_name_dict)
        
        arr = []
        add_to_arr(arr,decision_tree_dict)
        for item in arr:
            item.print_info()
        return {"message":arr,
                "error":"no"}
    except:
        return {"error":"yes"}


    





