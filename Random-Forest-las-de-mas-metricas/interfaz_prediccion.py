
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import pandas as pd

class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor de Características de Vacas")
        self.root.geometry("800x600")

        self.setup_ui()

    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        title_label = tk.Label(main_frame, text="Predictor de Características de Vacas", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # Frame para selección de archivo
        file_frame = tk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=10)

        self.file_label = tk.Label(file_frame, text="Ningún archivo seleccionado", width=50)
        self.file_label.pack(side=tk.LEFT, padx=(0, 10))

        browse_button = tk.Button(file_frame, text="Seleccionar Archivo PLY", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT)

        # Botón de predicción
        predict_button = tk.Button(main_frame, text="Realizar Predicción", command=self.predict, bg="#4CAF50", fg="white", height=2)
        predict_button.pack(pady=20)

        # Frame para resultados
        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Tabla de resultados
        self.create_results_table()

        # Historial
        history_label = tk.Label(main_frame, text="Historial de Predicciones", font=("Arial", 12, "bold"))
        history_label.pack(pady=(20, 10))

        self.history_listbox = tk.Listbox(main_frame, height=5)
        self.history_listbox.pack(fill=tk.X)
        self.history_listbox.bind("<<ListboxSelect>>", self.load_history_item)

        # Botón para exportar a Excel
        export_button = tk.Button(main_frame, text="Exportar Historial a Excel", command=self.export_to_excel)
        export_button.pack(pady=10)

    def create_results_table(self):
        # Limpiar frame anterior
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Cabeceras
        headers = ["Característica", "Valor", "Unidad"]
        for i, header in enumerate(headers):
            label = tk.Label(self.results_frame, text=header, font=("Arial", 10, "bold"))
            label.grid(row=0, column=i, padx=10, pady=5, sticky="w")

        # Separador
        separator = tk.Frame(self.results_frame, height=2, bg="black")
        separator.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)

        # Historial de resultados
        self.results = []

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Archivo PLY",
            filetypes=[("Archivos PLY", "*.ply")]
        )

        if file_path:
            self.file_label.config(text=os.path.basename(file_path))
            self.file_path = file_path

    def predict(self):
        if not hasattr(self, 'file_path'):
            messagebox.showerror("Error", "Por favor seleccione un archivo PLY primero")
            return

        try:
            # Ejecutar script de predicción
            result = subprocess.run(
                ["python", "/content/drive/MyDrive/modelos_multiatributos_20250508_0425/prediccion_multiatributos.py", self.file_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                messagebox.showerror("Error", f"Error en la predicción: {result.stderr}")
                return

            # Procesar salida
            output = result.stdout
            predictions = {}

            # Parsear resultados de la salida
            start_parsing = False
            for line in output.split('\n'):
                if "Resultados de la predicción:" in line:
                    start_parsing = True
                    continue

                if start_parsing and line.strip() and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] not in ["Característica", "Valor", "Unidad"]:
                        trait = parts[0]
                        value = float(parts[1])
                        unit = parts[2] if len(parts) > 2 else ""
                        predictions[trait] = (value, unit)

            # Mostrar resultados
            self.display_results(predictions)

            # Añadir a historial
            history_entry = f"{os.path.basename(self.file_path)} - {len(predictions)} características"
            self.history_listbox.insert(0, history_entry)
            self.results.insert(0, {
                'file': self.file_path,
                'predictions': predictions
            })

        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar la predicción: {str(e)}")

    def display_results(self, predictions):
        # Limpiar tabla
        self.create_results_table()

        # Mostrar resultados
        row = 2
        for trait, (value, unit) in sorted(predictions.items()):
            tk.Label(self.results_frame, text=trait).grid(row=row, column=0, padx=10, pady=2, sticky="w")
            tk.Label(self.results_frame, text=f"{value:.2f}").grid(row=row, column=1, padx=10, pady=2, sticky="w")
            tk.Label(self.results_frame, text=unit).grid(row=row, column=2, padx=10, pady=2, sticky="w")
            row += 1

    def load_history_item(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            item = self.results[index]
            self.display_results(item['predictions'])
            self.file_label.config(text=os.path.basename(item['file']))
            self.file_path = item['file']

    def export_to_excel(self):
        if not self.results:
            messagebox.showinfo("Información", "No hay datos para exportar")
            return

        try:
            # Preparar datos para excel
            data = []
            for result in self.results:
                row = {'Archivo': os.path.basename(result['file'])}
                for trait, (value, unit) in result['predictions'].items():
                    row[f"{trait} ({unit})"] = value
                data.append(row)

            # Crear dataframe
            df = pd.DataFrame(data)

            # Guardar a Excel
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                title="Guardar historial como"
            )

            if save_path:
                df.to_excel(save_path, index=False)
                messagebox.showinfo("Éxito", f"Datos exportados a {save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar datos: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
