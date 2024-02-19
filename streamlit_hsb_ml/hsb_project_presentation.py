import streamlit as st

def project_presentation():
     st.write(f"""
    <div class = 'all'>
        <h1>Project Presentation</h1>  
        <h2>Project Overview</h2>
        <p>
            This project aims at building an application that classifies wether user is at
              risk of having a cardiovascular disease or not, based on their clinical characteristics.
        </p>
        <h2>Objectives</h2>
        <ul>
            <li>To understand the relationships between clinical features and, particularly, their connection to the target feature through relevant visualisations.</li>
            <li>To find a suitable machine learning model for this application</li>
            <li>To evaluate the impact of hyperparameters on the model's performances</li>
            <li>To put the model into production so it can treat user's input</li>
        </ul>
        </p>
        <h2>Material</h3>
        <ul>
            <li>Dataset used in this project is the Cardiovascular Disease dataset,
              available on
              <a href = 'https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset'>Kaggle</a>.</li>
            <li>Libraries used include <b>Pandas</b>, <b>NumPy</b>, <b>Matplotlib</b>, 
             <b>Seaborn</b>, <b>SciPy</b> and <b>Sci-kit Learn</b>.</li>
            <li>The project is implemented in Python and designed as a Streamlit application, 
             with HTML and CSS used for formatting.</li>
        </ul>
        <h2>Methods</h2>
              <p>
              <strong>Data analysis</strong>: Objectives were to provide general understanding of the
              dataset, help in feature selection and draw clinically relevant insights on adult population
              with cardiovascular disease. Dataset consisted of 70000 observation with a roughly 50% rate
              of patients and controls.
              Results from this analysis are available in the Data Viz section.              
              </p>
              <p>
              <strong>Model Selection</strong>. Several models were evaluated on this dataset: Logistic
              Regression, Random Forest, Support Vector Machines, K-Nearest Neighbors and AdaBoost. 
              Support Vector Machines was selected for this project, as it presented the best accuracy 
              (73.2%), a good rate of correct predictions for Class 0 (77.6%) and an acceptable rate 
              for Class 1 (68.8%).               
              </p>
              <p>
              <strong>Hyperparameter Tuning</strong>. SVM model was tuned to improve its
              performances, several combinations of kernel, C and gamma were tested.
              RBF kernel, with C = 0.1 and gamma = 10 led to good and balanced performances, and was the best
              combination for a high recall without a significant loss in accuracy. 
              <br>Additionally, several tests were performed using some features combination. Models were also
              trained based on sex and on age groups. Unfortunately, these adjustments did not lead to significant
              improvement in the model, with accuracy and recall for both class hovering around 73%.
              </p>
              <p>
              <strong>Deployment</strong>. Final tuning on the model led to an accuracy of 73.6%.
              This model was deployed in the present streamlit application.               
              </p>
    </div>

            """, unsafe_allow_html=True)