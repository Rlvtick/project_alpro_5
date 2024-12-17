from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from forecasting import preprocess_data, train_and_forecast

app = Flask(__name__)
UPLOAD_FOLDER = './uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET"])
def landing_page():
    """
    Landing page.
    """
    return render_template("landing_page.html")

@app.route("/submission", methods=["GET", "POST"])
def upload_file():
    """
    Upload file page.
    """
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("error.html", message="No file selected!")
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return redirect(url_for("forecast_options", filename=file.filename))
    return render_template("upload.html")

@app.route("/forecast/<filename>", methods=["GET"])
def forecast_options(filename):
    """
    Forecast options page.
    """
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        data, columns = preprocess_data(filepath)
        columns = [col for col in columns if col != "overall"]  # Exclude "overall"
    except ValueError as e:
        return render_template("error.html", message=str(e))

    return render_template("forecast.html", columns=columns, file=filename)

@app.route("/get_plot", methods=["POST"])
def get_plot():
    """
    Generate forecast plot.
    """
    filename = request.form.get("file")
    horizon = request.form.get("horizon")
    variable = request.form.get("variable")

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        plot_html = train_and_forecast(filepath, variable, horizon)
        return jsonify({"success": True, "plot_html": plot_html})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)