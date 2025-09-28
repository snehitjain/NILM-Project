# Non-Intrusive Load Monitoring (NILM) Project

This is a complete NILM project with Indian dataset structure for testing.

tep-by-Step Guide
1. Download and Prepare the iAWE Dataset

Access the Dataset: Visit the iAWE dataset page
 and download the electricity.tar.gz file.

Extract the Data: Extract the contents of electricity.tar.gz.

Organize the Files: Place the extracted .csv files into the data/ directory of your project.

2. Install Required Libraries

Navigate to your project directory and install the necessary Python libraries:

pip install -r requirements.txt

3. Train the Model

Run the following script to train your model:

python src/train.py

4. Evaluate the Model

After training, evaluate the model's performance:

python src/evaluate.py

5. Visualize the Results

Generate visualizations to understand the disaggregation:

python src/visualize.py

6. Run the Streamlit Application

Start the Streamlit app to interact with the model:

streamlit run app/app.py


This will open a browser window displaying the appliance-level energy consumption predictions.
