# dadu-predictor

![dadu_grid](https://user-images.githubusercontent.com/15711032/162850873-5aeda80f-4e91-4da0-9999-d08888eed040.png)

Housing prices have created a crisis in the United States, stemming from policy and financial attitudes. With adapting policy being the simpler course of action, many architects and urban planners are looking to increase housing density to combat the issue of supply. However, land use policy is slow moving, so intermediate solutions are required. Allowing single family lots to contain a DADU in the rear yard helps to alleviate the density problem. Policy surrounding ADUs has begun to shift, primarily in west coast cities such as Seattle, and Los Angeles. 

This research involves the creation of a web-based application for predicting energy, carbon, and financial impacts of potential DADUs through the use of surrogate modeling. The purpose of such a tool is to enable both homeowners and designers estimates of performance to drive production of DADUs. The dataset was created by simulating the entire examined design space using Rhinoceros, Grasshopper, and Honeybee. Simulation data is stored in .CSV files which were then preprocessed through Pandas. XGBoost was selected as the machine learning option, with hyperparameter optimization via bayesian search. The web app was then constructed using Streamlit and hosted on Heroku.
