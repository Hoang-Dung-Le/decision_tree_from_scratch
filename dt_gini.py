import numpy as np
from collections import Counter

class DecisionTreeC45Entropy:
    def __init__(self, attribute_name_dict, continuous_attributes):
        self.buoc = 1
        self.attribute_name_dict = attribute_name_dict
        self.step = []
        self.continuous_attributes = continuous_attributes

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

            if self.attribute in self.continuous_attributes:
                if attribute_value <= self.threshold:
                    child_node = self.children["<="]
                else:
                    child_node = self.children[">"]
            else:
                child_node = self.children[attribute_value]

            return child_node.predict(sample)

    def entropy(self, y, value, pr=True):
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
        entropy = sum(p * np.log2(p) for p in probabilities)
        if entropy != 0:
          entropy = - entropy
        if pr:
          detail_stepp = "entroppy " + str(value) + " :" + str(round(entropy, 2))
          self.step.append(detail_stepp)
        return entropy

    def information_gain(self, X, y, attribute, threshold=None):
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
            subset_entropy += (len(subset_y) / len(y)) * self.entropy(subset_y, value)
        stp_if = "Entropy của thuộc tính " + str(attribute) + " :" + str(round(subset_entropy,2))
        self.step.append(stp_if)

        return self.entropy(y, value,pr=False) - subset_entropy

    def choose_best_attribute(self, X, y, used_attributes):
        attrs = ""
        for attribute in range(X.shape[1]):
            if attribute not in used_attributes:
                attrs += str(self.attribute_name_dict.get(attribute))
                attrs += " "
        # print("Buoc: ", self.buoc)
        # print("Tính gini cho các thuộc tính: ", attrs)
        step_buoc = "Bước " + str(self.buoc)
        step_gini = "Tính gini cho các thuộc tính: " + str(attrs)
        self.step.append(step_buoc)
        self.step.append(step_gini)
        best_gain = 0
        best_attribute = None
        best_threshold = None

        for attribute in range(X.shape[1]):
            if attribute not in used_attributes:
                step_att = "Tính entropy cho thuộc tính " + str(self.attribute_name_dict.get(attribute))
                self.step.append(step_att)
                if attribute in self.continuous_attributes:
                    values = set(X[:, attribute])
                    for value in values:
                        gain = self.information_gain(X, y, attribute, threshold=value)
                        if gain > best_gain:
                            best_gain = gain
                            best_attribute = attribute
                            best_threshold = value
                else:
                    gain = self.information_gain(X, y, attribute)
                    if gain > best_gain:
                        best_gain = gain
                        best_attribute = attribute

        # print("Best Attribute:", best_attribute)  # In best attribute
        step_best_attribute = "Chọn thuộc tính: " + str(self.attribute_name_dict.get(best_attribute))
        self.step.append(step_best_attribute)
        self.buoc += 1
        return best_attribute, best_threshold

    def create_decision_tree(self, X, y, used_attributes=None):
        if len(set(y)) == 1:
            # Nếu tất cả các nhãn giống nhau, tạo nút lá
            return self.Node(label=y[0])

        if X.shape[1] == 0 or (used_attributes is not None and len(used_attributes) == X.shape[1]):
            # Nếu không còn thuộc tính hoặc tất cả các thuộc tính đã được sử dụng, tạo nút lá với nhãn phổ biến nhất
            most_common_label = Counter(y).most_common(1)[0][0]
            return self.Node(label=most_common_label)

        if used_attributes is None:
            used_attributes = set()

        best_attribute, best_threshold = self.choose_best_attribute(X, y, used_attributes)

        if best_attribute is None:
            # Nếu không tìm thấy thuộc tính phù hợp, tạo nút lá với nhãn phổ biến nhất
            most_common_label = Counter(y).most_common(1)[0][0]
            return self.Node(label=most_common_label)

        node = self.Node(attribute=best_attribute, threshold=best_threshold)
        used_attributes.add(best_attribute)

        if best_attribute in self.continuous_attributes:
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
                    node.add_child(value, self.Node(label=most_common_label))
                else:
                    node.add_child(value, self.create_decision_tree(subset_X, subset_y, used_attributes))
        else:
            values = set(X[:, best_attribute])
            for value in values:
                subset_indices = X[:, best_attribute] == value
                subset_X = X[subset_indices]
                subset_y = y[subset_indices]

                if len(subset_X) == 0:
                    # Nếu không còn dữ liệu, tạo nút lá với nhãn phổ biến nhất
                    most_common_label = Counter(y).most_common(1)[0][0]
                    node.add_child(value, self.Node(label=most_common_label))
                else:
                    node.add_child(value, self.create_decision_tree(subset_X, subset_y, used_attributes))
        return node

    def predict_samples(self, X, tree):
        return [tree.predict(sample) for sample in X]