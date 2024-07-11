import tkinter as tk
from tkinter import ttk
from math import pow, sqrt
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
class RangeInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Graphical User Interface (GUI) for estimating compressive strength of foam composite concrete containing fly ash")
        master.configure(background="#f0f0f0")
        window_width = 825
        window_height = 760
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_cord = 0  # Start from the left edge of the screen
        y_cord = 0  # Start from the top edge of the screen
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        x_cord = 0
        y_cord = 0
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        main_heading = tk.Label(master, text="Graphical User Interface (GUI) for: \n Estimating compressive strength of foam composite concrete containing fly ash",
                               bg="black", fg="#FFFFFF", font=("Helvetica", 16, "bold"), pady=10)
        main_heading.pack(side=tk.TOP, fill=tk.X)
        self.content_frame = tk.Frame(master, bg="#E8E8E8")
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=50, anchor=tk.CENTER)
        self.canvas = tk.Canvas(self.content_frame, bg="#E8E8E8")
        self.scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#E8E8E8")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.input_frame.pack(side=tk.TOP, fill="both", padx=10, pady=10, expand=True)
        heading = tk.Label(self.input_frame, text="Input Parameters", bg="#FFFFFF", font=("Helvetica", 16, "bold"), padx=10, pady=10)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.output_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.output_frame.pack(side=tk.TOP, fill="both", padx=20, pady=20)
        heading = tk.Label(self.output_frame, text="Predictions", bg="#FFFFFF", fg="black", font=("Helvetica", 16, "bold"), padx=10, pady=10)
        heading.grid(row=0, column=0, columnspan=2, pady=10)
        self.input_frame.grid_columnconfigure(1, weight=1)
        self.input_frame.grid_columnconfigure(2, weight=1)
        self.create_entry("Sand:", 63, 7)
        self.create_entry("Foam:", 2, 9)
        self.create_entry("Age:", 14, 11)
        self.G6C9 = 9.26582660866559
        self.G1C8 = 5.64752721810041
        self.G2C9 = 651.885596433243
        self.G2C6 = -3.3995839791732
        self.G3C6 = -1.42317859968028
        self.G5C8 = 1.37559190145688
        self.G5C5 = 4.76272907157053
        self.G5C4 = 2.27649859773634
        self.G5C0 = 27.0390538983469
        self.G6C8 = -11.8589328092528
        self.G6C2 = -4.9369406441423
        self.G3C1 = 5.0755191773801
        self.G3C2 = -2.04929321906692
        self.G3C9 = -3.03659398059106
        self.G4C6 = 7.3504933086559
        self.G1C1 = -3.9763420924224
        self.G2C1 = -5.25463358309415
        self.G2C8 = 3.43175145100999
        self.create_entry("W/C:", 0.202222, 1)
        self.create_entry("Density:", 1200, 3)
        self.create_entry("Fly ash:", 8, 5)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.calculate_button_a = tk.Button(self.output_frame, text="Gene Expression Programming (GEP)", command=self.calculate_y_a,
                                          bg="red", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_a.grid(row=1, column=0, pady=10, padx=10)
        self.a_output_text_a = tk.Text(self.output_frame, height=2, width=30)
        self.a_output_text_a.grid(row=1, column=1, padx=10, pady=10)
        self.b_button_b = tk.Button(self.output_frame, text="Extreme Gradient Boosting (XGB)", command=self.calculate_b_b,
                                        bg="red", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.b_button_b.grid(row=2, column=0, pady=10, padx=10)
        self.b_output_text_b = tk.Text(self.output_frame, height=2, width=30)
        self.b_output_text_b.grid(row=2, column=1, padx=10, pady=10)
        developer_info = tk.Label(text="This GUI is developed by combined efforts of:\nMuhammad Saud Khan (khans28@myumanitoba.ca), University of Manitoba, Canada\nZohaib Mehmood (zoohaibmehmood@gmail.com), COMSATS University Islamabad, Pakistan",
                                  bg="green", fg="white", font=("Helvetica", 11, "bold"), pady=10)
        developer_info.pack()
    def create_entry(self, text, default_val, row):
        label = tk.Label(self.input_frame, text=text, font=("Helvetica", 12, "bold italic"), fg="darkred", bg="white", anchor="e")
        label.grid(row=row*2, column=1, padx=10, pady=5, sticky="e")
        entry = tk.Entry(self.input_frame, font=("Helvetica", 12), fg="darkgreen", bg="white", width=15, bd=1, relief=tk.GROOVE)
        entry.insert(0, f"{default_val:.5f}")
        entry.grid(row=row*2, column=2, padx=10, pady=5, sticky="w")
        setattr(self, f'entry_{row}', entry)
    def get_entry_values(self):
        try:
            d1 = float(self.entry_1.get())
            d2 = float(self.entry_3.get())
            d3 = float(self.entry_5.get())
            d4 = float(self.entry_7.get())
            d5 = float(self.entry_9.get())
            d6 = float(self.entry_11.get())            
            return d1, d2, d3, d4, d5, d6
        except ValueError as ve:
            return None
    def calculate_y_a(self):
        values = self.get_entry_values()
        if values is None:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5, d6 = values
        y = 0
        y += ((d5+(d5+d3))/(pow((d4+pow((pow(d1,2.0)+pow((((d5+d5)+self.G1C1)/sqrt(d2)),2.0)),2.0)),2.0)+self.G1C8))
        y += ((self.G2C8/((pow(((self.G2C9-pow(d5,2.0))+((d6/self.G2C6)*(d5+d3))),2.0)+sqrt(pow(d2,2.0)))/(d2+d2)))+self.G2C1)
        y += ((((d4+d2)/(self.G3C2*(self.G3C1+(d5/d6))))/self.G3C6)/(self.G3C1+((self.G3C9*((d4/self.G3C1)-d1))+d4)))
        y += ((d6-pow(pow(self.G4C6,2.0),2.0))/d2)
        y += (sqrt(d2)/((self.G5C8+(d6/((pow((d4/self.G5C0),2.0)-sqrt((d1/self.G5C4)))*(self.G5C4/(d6/d2)))))+(self.G5C5*d5)))
        y += ((d4/d2)*sqrt(sqrt(sqrt(pow(((self.G6C8-(d2*(pow((d4*self.G6C9),2.0)+(pow(d2,2.0)/self.G6C2))))*d5),2.0)))))
        self.a_output_text_a.delete(1.0, tk.END)
        self.a_output_text_a.insert(tk.END, f"{y:.2f}")
        self.a_output_text_a.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
    def calculate_b_b(self):
        values = self.get_entry_values()
        if values is None:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5, d6 = values
        try:
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\software-and-prediction-main\Composite foam"
            filename = r"GUI.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=500)
            regressor= MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=100,
                reg_lambda=0.1,
                gamma=1,
                max_depth=8
            ))
            model=regressor.fit(x_train, y_train)
            model= model.fit(x, y)
            y_pred=model.predict(x_train)
            y_pred=model.predict(x_test)
            input_data = np.array([d1, d2, d3, d4, d5, d6]).reshape(1, -1)
            y_pred = model.predict(input_data)
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, f"{y_pred[0][0]:.2f}")
            self.b_output_text_b.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
        except FileNotFoundError:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Excel file not found")
        except ValueError as ve:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Invalid data format")
        except Exception as e:
            self.b_output_text_b.delete(1.0, tk.END)
            self.b_output_text_b.insert(tk.END, "Error: Failure")
if __name__ == "__main__":
    root = tk.Tk()
    gui = RangeInputGUI(root)
    root.mainloop()