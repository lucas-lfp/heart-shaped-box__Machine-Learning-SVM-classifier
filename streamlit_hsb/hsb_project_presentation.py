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
             <b>Seaborn</b>, <b>SciPy</b>, <b>Sci-kit Learn</b>, <b>Joblib</b> and of course <b>Streamlit</b>.</li>
            <li>The project is implemented in Python and designed as a Streamlit application, 
             with HTML and CSS used for formatting.</li>
        </ul>
        <h2>Methods</h2>
              <p>
                Development of this application was made following a four-steps plan, as described below.
              </p>
              <p>
              <span class = 'method_step'>Data Analysis</span> - Objectives were to provide general understanding of the
              dataset, help in feature selection and draw clinically relevant insights on adult population
              with cardiovascular disease. Dataset consisted of <b>70000</b> observation with a roughly <b>50%</b> rate
              of patients and controls.
              Results from this analysis are available in the Data Viz section.              
              </p>
              <p>
              <span class = 'method_step'>Model Selection</span> - Several models were evaluated on this dataset: Logistic
              Regression, Random Forest, Support Vector Machines, K-Nearest Neighbors and AdaBoost. 
              <b>Support Vector Machines</b> was selected for this project, as it presented the best accuracy 
              (<b>73.2%</b>), a good rate of correct predictions for Class 0 (<b>77.6%</b>) and an acceptable rate 
              for Class 1 (<b>68.8%</b>).               
              </p>
              <p>
              <span class = 'method_step'>Hyperparameter Tuning</span> - SVM model was tuned to improve its
              performances, several combinations of kernel, C and gamma were tested.
              <code>RBF</code> kernel, with <code>C = 0.1</code> and <code>gamma = 10</code> led to good and balanced performances, and was the best
              combination for a high recall without a significant loss in accuracy. 
              <br>Additionally, several tests were performed using some features combination. Models were also
              trained based on sex and on age groups. Unfortunately, these adjustments did not lead to significant
              improvement in the model, with accuracy and recall for both class hovering around <b>73%</b>.
              </p>
              <p>
              <span class = 'method_step'>Deployment</span> - Final tuning on the model led to an accuracy of <b>73.6%</b>.
              This model was deployed in the present streamlit application.               
              </p>
    </div>

            """, unsafe_allow_html=True)