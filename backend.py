from fastapi import FastAPI, UploadFile, Form, File
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List
import numpy as np
from dt_entropy import DecisionTreeC45Entropy
from dt_gini import DecisionTreeGiniIndex
from fastapi import HTTPException
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
import mysql.connector
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import random

# database = databases.Database("postgresql://user:password@localhost/database_name")

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
UPLOAD_DIR = 'assets/public/datasets'
UPLOAD_DIR_JSON = 'assets/public/datasets_json'
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể chỉ định các nguồn cụ thể nếu cần
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory="assets/public"), name="static")


# db = mysql.connector.connect(
#     host="localhost",
#     port=3307,
#     user="root",
#     password="",
#     database="dt_web"
# )

# def login_check(username, password):
#     cursor = db.cursor()
#     query = "SELECT user_name FROM user WHERE user_name = %s and password = %s"
#     cursor.execute(query, (username, password,))
#     user = cursor.fetchone()
#     cursor.close()
#     if user:
#         return True
#     else:
#         return False


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
async def decision_tree_c45(file: Annotated[UploadFile, Form()], conti_attribute: Annotated[str, Form()], type: Annotated[str, Form()]):
    if file is None:
        return {"message": "No file received"}
    # Read the CSV file into a DataFrame
    # print(conti_attribute)
    # print(file.filename)
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
        if type == 'Entropy':
        # Tạo cây quyết định
            tree = DecisionTreeC45Entropy(attribute_name_dict, continuous_attributes)
            decision_tree = tree.create_decision_tree(X, y)
            decision_tree_dict = decision_tree_to_dict(decision_tree, attribute_name_dict)
            
            arr = []
            add_to_arr(arr,decision_tree_dict)
            steps = tree.get_step()
            # print(steps)
            # print(decision_tree_dict)
            print(tree.get_pratice())
            return {"message":arr,
                    "steps":steps,
                    "error":"no"}
        else:
            tree = DecisionTreeGiniIndex(attribute_name_dict, continuous_attributes)
            decision_tree = tree.create_decision_tree(X, y)
            decision_tree_dict = decision_tree_to_dict(decision_tree, attribute_name_dict)
            
            arr = []
            add_to_arr(arr,decision_tree_dict)
            steps = tree.get_step()
            # print(steps)
            # print(decision_tree_dict)
            print(tree.get_pratice())
            return {"message":arr,
                    "steps":steps,
                    "error":"no"}
    except:
        return {"error":"yes"}
    
# @app.post('/login')
# def login(user_name: Annotated[str, Form()], password: Annotated[str, Form()]):
#     print(user_name, password)
#     result = login_check(username=user_name, password=password)
#     if result:
#         return {"message": "ok"}
#     return {"message":"fail"}

# @app.get('/get_datasets')
# def get_datasets():
#     cursor = db.cursor()
#     query = "SELECT name, reliability FROM list_datasets"
#     cursor.execute(query, ())
#     list_datasets = cursor.fetchall()
#     cursor.close()
#     # print(list_datasets)
#     return {"message":list_datasets}

# @app.post('change-file')
# def change_file()


# @app.post("/upload_file")
# async def uploadFile(file: Annotated[UploadFile, Form()], conti_attribute: Annotated[str, Form()]):
#     if file is None:
#         return {"message": "No file received"}
#     # Read the CSV file into a DataFrame
#     # print(conti_attribute)

#     try:
#         global continuous_attributes
#         if conti_attribute == 'empty':
#             continuous_attributes = set()
#         else:
#             continuous_attributes = [int(num) for num in conti_attribute.split(",")]
#         # print(continuous_attributes)
#         data = pd.read_csv(file.file)
#         # print(data.head())
#         attribute_name = data.columns.values.tolist()
#         attribute_name_dict = {}
#         for i in range(len(attribute_name) - 1):
#             attribute_name_dict.update({i: attribute_name[i]})
#         # print(attribute_name_dict)
#         # data = data.iloc[1:]

#         # Chuyển đổi dữ liệu thành mảng numpy
#         data_np = np.array(data)

#         # Tách thuộc tính và nhãn
#         X = data_np[:, :-1]  # Thuộc tính
#         y = data_np[:, -1]  # Nhãn

#         # Tạo cây quyết định
#         tree = DecisionTreeC45Entropy(attribute_name_dict, continuous_attributes)
#         decision_tree = tree.create_decision_tree(X, y)
#         decision_tree_dict = decision_tree_to_dict(decision_tree, attribute_name_dict)
        
#         arr = []
#         add_to_arr(arr,decision_tree_dict)
#         steps = tree.get_step()
#         # print(steps)
#         # print(decision_tree_dict)
#         # print(tree.get_pratice())
#         test_case = tree.get_pratice()
#         file_path = os.path.join(UPLOAD_DIR, file.filename)
#         file_name, file_extension = os.path.splitext(file_path)
#         file_path_txt = file_name + ".txt"
#         with open(file_path_txt, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         file_name = str(uuid.uuid4()) + '.json'
#         file_path = os.path.join(UPLOAD_DIR_JSON, file_name)
#         with open(file_path, 'w', encoding='utf-8') as json_file:
#             json.dump(test_case, json_file, ensure_ascii=False)
#         cursor = db.cursor()
#         query = "INSERT INTO list_datasets (name, url, url_json) values(%s, %s, %s)"
#         cursor.execute(query, (file_path_txt, file_path_txt, file_name))
#         db.commit(   )
#         return {"message":arr,
#                 "steps":steps,
#                 "error":"no"}
#     except:
#         return {"error":"yes"}
    

# @app.get('/test_case')
# def get_test_case():
#     cursor = db.cursor()
#     query = "SELECT * FROM list_datasets"
#     cursor.execute(query, ())
#     list_datasets = cursor.fetchall()
#     cursor.close()
#     random_number = random.randint(0, len(list_datasets) - 1)
#     print(list_datasets[random_number][3])
#     file_json_name = UPLOAD_DIR_JSON + "/" + str(list_datasets[random_number][3])
#     with open(file_json_name, encoding='utf-8') as json_file:
#         data = json.load(json_file)
#     # print(data)
#     link = "http://localhost:8000/public/datasets/" + str(list_datasets[random_number][2])
#     result = data.get("1")
#     return {"message": result,
#             "link_dataset":link}


