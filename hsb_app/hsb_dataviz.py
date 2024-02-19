import streamlit as st

def data_viz():
    """
    Function to display content for the Data Viz section.
    """
    st.write("Welcome to the Data Viz section!")

    # Nested navigation within the "Data Viz" section
    feature_selection = st.sidebar.selectbox("Select Feature", ["Feature 1", "Feature 2", "Feature 3"])

    # Display content based on selected feature
    if feature_selection == "Feature 1":
        st.write("Displaying Feature 1...")
        # Call function to display Feature 1 visualization
    elif feature_selection == "Feature 2":
        st.write("Displaying Feature 2...")
        # Call function to display Feature 2 visualization
    elif feature_selection == "Feature 3":
        st.write("Displaying Feature 3...")
        # Call function to display Feature 3 visualization