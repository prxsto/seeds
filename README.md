# SEEDS

![code size](https://img.shields.io/github/languages/code-size/prxsto/seeds?style=flat-square)
![license](https://img.shields.io/github/license/prxsto/seeds?style=flat-square)
![issues](https://img.shields.io/github/issues/prxsto/seeds)

SEEDS (Seattle Economic and Environmental Dwelling Simulator) is a webtool for predicting energy, carbon, and associated costs of DADUs (Detached Accessory Dwelling Units) in Seattle, WA.

## Contributing

Feel free to create an issue/PR if you want to see anything else implemented or find any bugs. Streamlit apps can be hosted on a generic url for free while testing.

## Background

This project was developed alongside my M.S. thesis at the University of Washington, exploring accessory dwelling units as an architectural response to soaring housing prices. Thesis paper can be found [here](https://digital.lib.washington.edu/researchworks/handle/1773/48656).

![dadu_grid](https://user-images.githubusercontent.com/15711032/162850975-3552f1c9-fe26-4948-b1ad-d573be169326.png)

Housing prices have created a crisis in the United States, stemming from policy and financial attitudes. With adapting policy being the simpler course of action, many architects and urban planners are looking to increase housing density to combat the issue of supply. However, land use policy is slow moving, so intermediate solutions are required. Allowing single family lots to contain a DADU in the rear yard helps to alleviate the density problem. Policy surrounding ADUs has begun to shift, primarily in west coast cities such as Seattle, and Los Angeles. 

This research involves the creation of a web-based application for predicting energy, carbon, and financial impacts of potential DADUs through the use of surrogate modeling. The purpose of such a tool is to enable both homeowners and designers estimates of performance to drive production of DADUs. The dataset was created by simulating the entire examined design space using Rhinoceros, Grasshopper, and Honeybee. Simulation data is stored in .CSV files which were then preprocessed through Pandas. XGBoost was selected as the machine learning option, with hyperparameter optimization via bayesian search. The web app was then constructed using Streamlit and hosted on Heroku.

## Caveats

Keep in mind this regression model was trained using a very small subset of design constraints. Results from the tool can not be evaluated for conditions which were not simulated. Web application uses cost and carbon data from 2022 (TODO).
