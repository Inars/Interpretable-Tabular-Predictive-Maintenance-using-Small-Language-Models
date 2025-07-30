import pandas as pd
import json
from google import genai
from tqdm import tqdm

from utils import DATASETS, GOOGLE_API_KEY

class DataSerializer:
    """
    A class that serializes an input pandas DataFrame (data) to various formats.
    Each method returns a list of serialized items, one per row in the DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the serializer with a pandas DataFrame.
        
        :param data: A pandas DataFrame to be serialized.
        """
        self.data = data

    def to_csv(self) -> list:
        """
        Serialize each row to a CSV string.

        :return: A list, where each element is a CSV string representing a row.
        """
        serialized_list = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Serializing to CSV"):
            item = {}
            item ["ground_truth"] = str(row.iloc[-1])
            row = row[:-1]
            col_value_pairs = [f"{col}={row[col]}" for col in self.data.columns[:-1]]
            item["prompt"] = ",".join(col_value_pairs)
            serialized_list.append(item)
        return serialized_list

    def to_llm(self) -> list:
        """
        Generate a response using the Google GenAI API.

        :return: The generated text response from the API.
        """
        client = genai.Client(api_key=GOOGLE_API_KEY,)

        serialized_list = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Serializing to LLM"):
            item = {}
            item ["ground_truth"] = str(row.iloc[-1])
            item["prompt"] = client.models.generate_content(
                model="gemma-3-27b-it",
                contents="Write the following dictionary data into a cohesive and coherent paragraph: " + str(row),
            ).text
            serialized_list.append(item)
        return serialized_list

    def to_markdown(self) -> list:
        """
        Serialize each row to a small markdown table of key-value pairs.

        :return: A list, where each element is a markdown table for a single row.
        """
        serialized_list = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Serializing to Markdown"):
            item = {}
            item["ground_truth"] = str(row.iloc[-1])
            row = row[:-1]
            md_string = "| Column | Value |\n|--------|-------|\n"
            for col in row.index:
                md_string += f"| {col} | {row[col]} |\n"
            item["prompt"] = md_string
            serialized_list.append(item)
        return serialized_list

    def to_json(self) -> list:
        """
        Serialize each row to a JSON string.

        :return: A list, where each element is a JSON string representing a row.
        """
        serialized_list = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Serializing to JSON"):
            item = {}
            item["ground_truth"] = str(row.iloc[-1])
            row = row[:-1]
            row_dict = row.to_dict()
            item["prompt"] = json.dumps(row_dict)
            serialized_list.append(item)
        return serialized_list

    def to_list(self) -> list:
        """
        Serialize each row to a Python list (row values as a list).

        :return: A list, where each element itself is a list of row values.
        """
        serialized_list = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Serializing to List"):
            item = {}
            item["ground_truth"] = str(row.iloc[-1])
            row = row[:-1]
            row_values = list(row.values)
            item["prompt"] = str(row_values)
            serialized_list.append(item)
        return serialized_list

    def to_phrase(self) -> list:
        """
        Serialize each row to a hard-coded type phrase, for example:
        "For column c, we have value v. For column c2, we have value v2."

        :return: A list, where each element is a descriptive string for that row.
        """
        serialized_list = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Serializing to Phrase"):
            item = {}
            item["ground_truth"] = str(row.iloc[-1])
            row = row[:-1]
            phrase_parts = [
                f"For column '{col}', we have value '{value}'."
                for col, value in row.items()
            ]
            item["prompt"] = " ".join(phrase_parts)
            serialized_list.append(item)
        return serialized_list

def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit_number')
    max_time_cycles = train_grouped_by_unit['time_cycles'].max()
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1)
    return merged

def main():
    for task, dataset in DATASETS.items():
        print(f"Task: {task}")
        
        # Load the dataset
        if task == "binary" or task == "multiclass":
            df = pd.read_csv("./data/" + dataset + ".csv")

            # Convert temperature from Kelvin to Celsius
            df["Air temperature [K]"] = df["Air temperature [K]"] - 272.15
            df["Process temperature [K]"] = df["Process temperature [K]"] - 272.15

            df.rename(columns={"Air temperature [K]" : "Air temperature [°C]","Process temperature [K]" : "Process temperature [°C]"},inplace=True)

            if task == "binary":
                df = df.drop(columns=["Failure Type"])
            elif task == "multiclass":
                df = df.drop(columns=["Target"])
        elif task == "regression":
            index_names = ['unit_number', 'time_cycles']
            setting_names = ['setting_1', 'setting_2', 'setting_3']
            sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
            col_names = index_names + setting_names + sensor_names

            dftrain = pd.read_csv(f"./data/train_{dataset}.txt",sep='\s+',header=None,index_col=False,names=col_names)
            dfvalid = pd.read_csv(f"./data/test_{dataset}.txt",sep='\s+',header=None,index_col=False,names=col_names)
            y_valid = pd.read_csv(f"./data/RUL_{dataset}.txt",sep='\s+',header=None,index_col=False,names=['RUL'])

            dftrain = add_RUL_column(dftrain)
            
            dfvalid = pd.concat([dfvalid, y_valid], axis=1) 
            
            dftrain = pd.concat([dftrain, dfvalid], ignore_index=True)

            Sensor_dictionary={}
            dict_list=[ "Fan inlet temperature (◦R)",
            "LPC outlet temperature (◦R)",
            "HPC outlet temperature (◦R)",
            "LPT outlet temperature (◦R)",
            "Fan inlet Pressure (psia)",
            "bypass-duct pressure (psia)",
            "HPC outlet pressure (psia)",
            "Physical fan speed (rpm)",
            "Physical core speed (rpm)",
            "Engine pressure ratio (P50/P2)",
            "HPC outlet Static pressure (psia)",
            "Ratio of fuel flow to Ps30 (pps/psia)",
            "Corrected fan speed (rpm)",
            "Corrected core speed (rpm)",
            "Bypass Ratio",
            "Burner fuel-air ratio",
            "Bleed Enthalpy",
            "Required fan speed",
            "Required fan conversion speed",
            "High-pressure turbines Cool air flow",
            "Low-pressure turbines Cool air flow" ]
            i=1
            for x in dict_list :
                Sensor_dictionary['s_'+str(i)]=x
                i+=1

            dftrain.rename(columns=Sensor_dictionary, inplace=True)
            df = dftrain.head(10000) # Assign dftrain to df for further processing by DataSerializer

        serializer = DataSerializer(df)

        print(f"Serializing # {len(serializer.data)} rows of data...")

        # Serialize data to csv
        print("Serializing data to csv...")
        csv_output = serializer.to_csv()         # List of CSV strings
        df = pd.DataFrame(csv_output)
        df.to_csv("data/serialized/" + task + "/" + dataset + "_csv.csv", index=False)

        # Serialize data to makdown
        print("Serializing data to markdown...")
        md_output = serializer.to_markdown()       # List of markdown tables
        df = pd.DataFrame(md_output)
        df.to_csv("data/serialized/" + task + "/" + dataset + "_md.csv", index=False)

        # Serialize data to json
        print("Serializing data to json...")
        json_output = serializer.to_json()         # List of JSON strings
        df = pd.DataFrame(json_output)
        df.to_csv("data/serialized/" + task + "/" + dataset + "_json.csv", index=False)

        # Serialize data to list
        print("Serializing data to list...")
        list_output = serializer.to_list()         # List of row-lists
        df = pd.DataFrame(list_output)
        df.to_csv("data/serialized/" + task + "/" + dataset + "_list.csv", index=False)

        # Serialize data to phrase
        print("Serializing data to phrase...")
        phrase_output = serializer.to_phrase()     # List of descriptive phrases
        df = pd.DataFrame(phrase_output)
        df.to_csv("data/serialized/" + task + "/" + dataset + "_phrase.csv", index=False)

        # Serialize data to LLM phrases
        print("Serializing data to LLM phrases...")
        llm_output = serializer.to_llm()           # List of LLM phrases
        df = pd.DataFrame(llm_output)
        df.to_csv("data/serialized/" + task + "/" + dataset + "_llm.csv", index=False)

if __name__ == "__main__":
    main()
