import numpy as np
import matplotlib.pyplot as plt

def trapezoidal(a, b, c, d):
    if a == b:
        return lambda x: np.maximum(0, np.minimum(1, (d - x) / (d - c)))
    if c == d:
        return lambda x: np.maximum(0, np.minimum((x - a) / (b - a), 1))
    return lambda x: np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))

def triangular(a, b, c):
    return lambda x: trapezoidal(a, b, b, c)(x)
    # return lambda x: np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

def gaussian(a, b, c):
    return lambda x: 1 / (1 + (((x - c)/ a) ** (2 * b)))


class Variable:
    def __init__(self, name):
        self.name = name
        self.sets = []
        self.functions = {}
    
    def add_triangular(self, set_name, a, b, c):
        self.sets.append(set_name)
        self.functions[set_name] = triangular(a, b, c)

    def add_trapez(self, set_name, a, b, c, d):
        self.sets.append(set_name)
        self.functions[set_name] = trapezoidal(a, b, c, d)
    
    def plot(self, xmin, xmax):
        x = np.arange(xmin, xmax, 0.1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(self.name)
        for _set in self.sets:
            y = self(_set, x)
            ax.plot(x, y)
        plt.show()

    def __call__(self, fuzzySet, crispValue):
        return self.functions[fuzzySet](crispValue)


class Controller:
    def __init__(self, name, inputs, output):
        self.name = name

        self.inputs = {x.name : x for x in inputs}
        self.input_names = [x.name for x in inputs]

        self.output = output
        self.output_name = output.name

        self.rules = []
        
    def add_rule(self, rule):
        self.rules.append(rule)
    
    def fuzzify(self, values):
        result = {}
        for rule in self.rules:
            key = rule["out"][self.output_name]
            if key in result.keys():
                result[key] = max(result[key], self.compute(rule, values))
            else:
                result[key] = self.compute(rule, values)
        return result


    def plot(self, xmin, xmax, fuzzyValues):
        x = np.arange(xmin, xmax, 0.1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(self.output.name + " = " + str(self.defuzzify(xmin, xmax, fuzzyValues, 0.001)))
        for _set in self.output.sets:
            y = self.output(_set, x)
            ax.plot(x, y)
        y = self.defuzzifyFunction(x, fuzzyValues)
        ax.plot(x, y, color="black")
        plt.show()

    def defuzzifyFunction(self, x, fuzzyValues):
        values = np.zeros(x.shape)
        for _set in self.output.sets:
            if _set in fuzzyValues.keys():
                values = np.maximum(values, np.minimum(fuzzyValues[_set], self.output(_set, x)))
        return values

    def defuzzify(self, xmin, xmax, fuzzyValues, w=0.01):
        # Find the defuzzifiedValue of the function
        x = np.arange(xmin, xmax, w)
        y = self.defuzzifyFunction(x, fuzzyValues)

        centroid = 0
        areas = 0
        for _x, _y in zip(x, y):
            area = w * _y
            centroid += ((_x + (w / 2)) * area)
            areas += area

        return centroid / areas

    def calculate(self, rule, outerKey, values):
        fuzzyValues = []
        for key in rule.keys():
            if key in ["or", "and"]:
                fuzzyValues.append(self.compute(rule[key], key, values))
            else:
                crispValue = values[key]
                fuzzySet = rule[key]
                fuzzyValues.append(self.inputs[key](fuzzySet, crispValue))
                
        if outerKey == "or":
            return max(fuzzyValues)
        else:
            return min(fuzzyValues)


    def compute(self, rule, values):
        keys = list(rule["in"].keys())
        if len(keys) == 1:
            if keys[0] == "or":
                return self.calculate(rule["in"]["or"], "or", values)
            else:
                return self.calculate(rule["in"]["and"], "and", values)


if __name__ == "__main__":
    # acceleration = Variable("acceleration")
    # acceleration.add_trapez("NL", 0, 0, 31, 61)
    # acceleration.add_triangular("NM", 31, 61, 95)
    # acceleration.add_triangular("NS", 61, 95, 127)
    # acceleration.add_triangular("ZE", 95, 127, 159)
    # acceleration.add_triangular("PS", 127, 159, 191)
    # acceleration.add_triangular("PM", 159, 191, 223)
    # acceleration.add_trapez("PL", 191, 233, 255, 255)

    # speedDifference = Variable("speedDifference")
    # speedDifference.add_trapez("NL", 0, 0, 31, 61)
    # speedDifference.add_triangular("NM", 31, 61, 95)
    # speedDifference.add_triangular("NS", 61, 95, 127)
    # speedDifference.add_triangular("ZE", 95, 127, 159)
    # speedDifference.add_triangular("PS", 127, 159, 191)
    # speedDifference.add_triangular("PM", 159, 191, 223)
    # speedDifference.add_trapez("PL", 191, 233, 255, 255)

    # throttle = Variable("throttle")
    # throttle.add_trapez("NL", 0, 0, 31, 61)
    # throttle.add_triangular("NM", 31, 61, 95)
    # throttle.add_triangular("NS", 61, 95, 127)
    # throttle.add_triangular("ZE", 95, 127, 159)
    # throttle.add_triangular("PS", 127, 159, 191)
    # throttle.add_triangular("PM", 159, 191, 223)
    # throttle.add_trapez("PL", 191, 233, 255, 255)


    # c = Controller("Throttle", [acceleration, speedDifference], throttle)

    # #Rule 1 If (speed difference is NL) and (acceleration is ZE) then (throttle control is PL)
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "ZE",
    #             "speedDifference" : "NL",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "PL"
    #     }
    # })

    # #Rule 2 If (speed difference is ZE) and (acceleration is NL) then (throttle control is PL)
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "NL",
    #             "speedDifference" : "ZE",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "PL"
    #     }
    # })

    # #Rule 3 If (speed difference is NM) and (acceleration is ZE) then (throttle control is PM)
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "ZE",
    #             "speedDifference" : "NM",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "PM"
    #     }
    # })

    # #Rule 4 If (speed difference is NS) and (acceleration is PS) then (throttle control is PS)
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "PS",
    #             "speedDifference" : "NS",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "PS"
    #     }
    # })

    # #Rule 5 If (speed difference is PS) and (acceleration is NS) then (throttle control is NS) 
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "NS",
    #             "speedDifference" : "PS",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "NS"
    #     }
    # })

    # #Rule 6 If (speed difference is PL) and (acceleration is ZE) then (throttle control is NL)
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "ZE",
    #             "speedDifference" : "PL",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "NL"
    #     }
    # })

    # #Rule 7 If (speed difference is ZE) and (acceleration is NS) then (throttle control is PS)
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "NS",
    #             "speedDifference" : "ZE",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "PS"
    #     }
    # })

    # #Rule 8 If (speed difference is ZE) and (acceleration is NM) then (throttle control is PM)
    # c.add_rule({
    #     "in" : {
    #         "and" : {
    #             "acceleration" : "NM",
    #             "speedDifference" : "ZE",
    #         }
    #     },
    #     "out" : {
    #         "throttle" : "PM"
    #     }
    # })

    # values = c.fuzzify({"acceleration" : 70, "speedDifference" : 100})
    # print(values)
    # c.plot(0, 300, values)

    service = Variable("service")
    service.add_trapez("poor", 0, 0, 0, 5)
    service.add_triangular("average", 0, 5, 10)
    service.add_trapez("good", 5, 10, 10, 10)

    quality = Variable("quality")
    quality.add_trapez("poor", 0, 0, 0, 5)
    quality.add_triangular("average", 0, 5, 10)
    quality.add_trapez("good", 5, 10, 10, 10)

    tip = Variable("tip")
    tip.add_trapez("low", 0, 0, 0, 12)
    tip.add_triangular("medium", 0, 12, 25)
    tip.add_trapez("high", 12, 25, 25, 25)

    c = Controller("tip", [service, quality], tip)
    
    c.add_rule({
        "in" : {
            "or" : {
                "service" : "good",
                "quality" : "good"
            }
        },
        "out" : {
            "tip" : "high"
        }
    })

    c.add_rule({
        "in" : {
            "or" : {
                "service" : "average",
            }
        },
        "out" : {
            "tip" : "medium"
        }
    })

    c.add_rule({
        "in" : {
            "and" : {
                "service" : "poor",
                "quality" : "poor",
            }
        },
        "out" : {
            "tip" : "low"
        }
    })

    quality.plot(0, 11)
    service.plot(0, 11)
    tip.plot(0, 26)

    value = c.fuzzify({"service" : 9.8, "quality" : 6.5})
    c.plot(0, 26, value)