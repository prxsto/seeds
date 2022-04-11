import os
from re import X
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import xgboost as xgb
import datetime
import make_mesh
from math import sqrt

# dictionaries to translate between user input and prediction input values
infd = {
    'Typical': 0.00059,
    'Passive house': 0.00015
}
oriend = {
    'North': 1,
    'South': 2,
    'East': 3,
    'West': 4
}
setbackd = {
    'Existing': 0,
    'Proposed': 1
}
typologyd = {
    '1 unit, 1 story': {'num_units': 1, 'num_stories': 1},
    '1 unit, 2 stories': {'num_units': 1, 'num_stories': 2},
    '2 units, 1 story': {'num_units': 2, 'num_stories': 1}
}
sited = {
    'Corner with alley':0,
    'Corner without alley':1,
    'Infill with alley':2,
    'Infill without alley':3
}
plotd = {
    'Lot type': 'site',
    'Infiltration rate': 'inf_rate',
    'Orientation': 'orientation',
    'Setbacks': 'setback',
    'Floor area': 'size',
    'WWR': 'wwr',
    'R-assembly': 'assembly_r',
    'EUI': 'eui_kbtu',
    'CO2': 'annual_carbon',
    'Cost': 'annual_cost'
}
durationd = {
    '1 month': 1/12,
    '6 months': .5,
    '1 year': 1,
    '5 years': 5,
    '10 years': 10,
}

# model is cached to avoid unpickling model each prediction; further, hashing is disabled per Streamlit documentation
@st.cache(hash_funcs={xgb.core.Booster: lambda _: None})
def load_model():
    """ 
    Loads pickled xgboost model to predict EUI
    """
    # os.chdir(r'/Users/prxsto/Documents/GitHub/msdc-thesis')
    pickle_in = open('xgboost_reg.pkl', 'rb')
    regressor = pickle.load(pickle_in)
    return regressor
    
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
 
def calc_r(polyiso_t, cellulose_t):
    """
    Calculates wall assembly R value.

    Args:
        polyiso_t (float): thickness of polyiso insulation (inches)
        cellulose_t (float): thickness of cellulose insulation (inches)

    Returns:
        assembly_r (float): wall assembly R value
    """
    air_gap = 1.0
    cladding = 0.61 #aluminum
    ply = 0.63
    ext_air_film = .17
    int_air_film = .68
    
    assembly_r = ext_air_film + cladding + air_gap + polyiso_t*7 + ply + cellulose_t*3.5 + int_air_film
    
    return assembly_r

def create_input_df(site, size, footprint, height, num_stories, num_units, inf_rate, orientation, wwr, setback, assembly_r, surf_vol_ratio):
    """
    Takes user input from Streamlit sliders and creates 1D dictionary from variables, then converts to DataFrame

    Args:
        site (int): site to simulate, currently 0-3 and modeled in Grasshopper
        size (int): total square footage of DADU
        footprint (int): first floor square footage of DADU; footprint == size for 1 story DADU
        height (int): height of DADU; 10 for 1 story, 20 for 2
        num_stories (int): number of stories (1,2)
        num_units (int): number of units (1,2); cannot have 2 units in 2 story DADU
        inf_rate (float): infiltration rate (standard or passive house)
        orientation (int): orientation of existing house to DADU (0,1,2,3; N,S,E,W)
        wwr (float): window-to-wall ratio (0.0<=1.0)
        setback (int): existing or revised, lax lot setbacks
        assembly_r (float): total r value of wall assembly
        surf_vol_ratio (float): ratio of total surface area to volume

    Returns:
        pred_input (DataFrame): dataframe of shape (1,15)
    """
    inputs = {
        'site': [site], 'size': [size], 'footprint': [footprint], 'height': [height], 'num_stories': [num_stories], 'num_units': [num_units],
        'inf_rate': [inf_rate], 'orientation': [orientation], 'wwr': [wwr], 'setback': [setback], 'assembly_r': [assembly_r], 
        'surf_vol_ratio': [surf_vol_ratio]
    }
    pred_input = pd.DataFrame(inputs)
    return pred_input

def predict_eui(pred_input, model):
    """Predicts energy use intensity of DADU from user input.

    Args:
        pred_input (DataFrame): DataFrame of shape (1,15) containing user input for prediction

    Returns:
        prediction (float): energy use intensity (EUI; kBTU/ft^2)
    """
    pred_inputDM = xgb.DMatrix(pred_input)
    prediction = model.predict(pred_inputDM)
    return prediction

def user_favorites(results, count, favorites): #TODO
    """
    Takes user's list of favorites appends results of each to new dataframe. Allows user to download csv 
    with all of their top picks.

    Args:
        full_df (DataFrame): DataFrame containing results of favorited simulation
        favorites (List): List containing all of the runs which are flagged as favorites
    """
    fav_df = pd.DataFrame()
    for row in favorites:
        df = df.append(row, ignore_index=True)
    return fav_df

def percent_change(old, new):
    """
    Calculates percent change. Used within tool to show changes between prediction results.

    Args:
        old (float): Value of previous result
        new (float): Value of current results

    Returns:
        pc (float): Percent change between values
    """
    pc = round((new - old) / abs(old) * 100, 2)
    return pc

def plot_scatter(x, y, color, x_axis_data, y_axis_data):
    """
    Creates and plots Plotly scatter with prediction data.

    Args:
        x (series): Column of data from results DataFrame chosen by user in drop-down menu to display on X axis
        y (series): Column of data from results DataFrame chosen by user in drop-down menu to display on Y axis
        color (series): Column of data from results DataFrame chosen by user in drop-down menu to color points by
        x_axis_data (string): Column name of X axis data
        y_axis_data (string): Column name of Y axis data

    Returns:
        fig (Plotly figure): Scatterplot showing user-selected prediction data
    """
    if y_axis_data == 'Cost':
        y_axis_data = 'Annual Cost ($)'
        hover = 'Cost: $%{y}<extra></extra>'
    if y_axis_data == 'CO2':
        y_axis_data = 'Annual Carbon (kgCO2)'
        hover = 'Carbon: %{y} kgCO2<extra></extra>'
    if y_axis_data == 'EUI':
        y_axis_data = 'EUI (kBTU/ft2)'
        hover = 'EUI: %{y} kBTU/ft2<extra></extra>'
    if x_axis_data == 'R-assembly':
        x_axis_data = 'R-assembly (ft2·°F·h/BTU)'
    if x_axis_data == 'Infiltration rate':
        x_axis_data = 'Infiltration rate (m3/s per m2 of facade)'
    if x_axis_data == 'Floor area':
        x_axis_data = 'Floor area (ft2)'
    
    # if x_axis_data == 'Lot type':
    #     bins = pd.interval_range(start=0, end=4)
    #     d = dict(zip(bins, ['Corner/alley', 'Corner/no alley', 'Infill/alley', 'Infill/no alley']))
    #     pd.cut(x, bins).map(d)
    #     st.write(x)
    # if x_axis_data == 'Orientation':
    #     for i in x:
    #         if i == 0:
    #             i = 'N'
    #         if i == 1:
    #             i = 'S'
    #         if i == 2:
    #             i = 'E'
    #         if i == 3:
    #             i = 'W'

    # if x_axis_data == 'Setbacks':
    #     for i in x:
    #         if i == 0:
    #             i = 'Existing'
    #         if i == 1:
    #             i = 'Proposed'
                           
    scatter = go.Scattergl(x=x, 
                        y=y,
                        marker_color=color,
                        text=color,
                        mode='markers',
                        hovertemplate=hover,
                        marker= {
                            'size': 12,
                            'colorscale': 'Viridis',
                            'showscale': True
                        }
                        )
    fig = go.Figure(data=scatter)
    
    fig.update_xaxes(title_text=x_axis_data)
    fig.update_yaxes(title_text=y_axis_data)
    if x_axis_data == 'Lot type' or x_axis_data == 'Orientation' or x_axis_data == 'Setbacks':
        fig.update_xaxes(type='category')
    fig.update_layout(hovermode='closest',
                        clickmode='event',
                        margin={'pad':10,
                                'l':50,
                                'r':50,
                                'b':50,
                                't':50},
                        font=dict(
                            size=18,
                            color="black")
                        )
    return fig
    
def web_tool(model):

    # hide streamlit logo
    hide_streamlit_logo = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_logo, unsafe_allow_html=True)
    
    # increase sidebar width
    increase_sidebar_width = """
        <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {width: 350px;}
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {width: 350px; margin-left: -350px;}
        </style>
        """
    st.markdown(increase_sidebar_width, unsafe_allow_html=True)
    
    # reduce margins of sidebar and canvas
    reduce_margins = """
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-zbg2rx {
                    padding-top: 2rem;
                    padding-right: 1rem;
                    padding-bottom: 2rem;
                    padding-left: 1rem;
                }
                .css-1j6homm {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
        </style>
        """
    st.markdown(reduce_margins, unsafe_allow_html=True)

    # hide collapse button
    hide_collapse = """
        <style>
            .css-119ihf6 {visibility: hidden;}
        <style>    
        """
    st.markdown(hide_collapse, unsafe_allow_html=True)
    
    st.title('DADU Impact Predictor')    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader('Results')
    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame()
        
    count = len(st.session_state.results.index)
    
    # constants
    kgCO2e = .135669
    kwh_cost = .1189
    mshp_cop = 3.5 # average COP value of mini split heat pump systems in use in most DADUs in PNW
    
    if count >= 1:
        rounded_eui = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('eui_kbtu')]
        rounded_eui_kwh = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('eui_kwh')]
        rounded_co2 = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('annual_carbon')]
        rounded_cost = st.session_state.results.iat[count - 1, st.session_state.results.columns.get_loc('annual_cost')]
    
    # create sidebar form
    with st.sidebar.form(key='user_input'):
        # sidebar dropdowns
        site = st.selectbox('Lot type', options=['Corner with alley', 'Corner without alley', 
                                                     'Infill with alley', 'Infill without alley'], 
                                index=3, help='Select the type of lot that your existing dwelling belongs to')
        typology = st.selectbox('DADU typology', ['1 unit, 1 story', '1 unit, 2 stories', 
                                        '2 units, 1 story'], index=0, 
                                        help='Select the number of stories and units')
        inf_rate = infd[st.selectbox('Infiltration rate (m3/s per m2 of facade)', 
                                             ['Typical', 'Passive house'], 
                                             help='Select either standard infiltration rate or passive house (extremely tight enclosure)',
                                             index=0, key='inf')]
        orientation = oriend[st.selectbox('Orientation',
                                                  ['North', 'South', 'East', 'West'], 
                                                  help='Select the direction of existing dwelling to DADU',
                                                  key='orientation')]
        setback = setbackd[st.selectbox('Land use setbacks',
                                                ['Existing', 'Proposed'], 
                                                help='Select either existing (2022) or proposed (more lenient) setbacks',
                                                key='setback')] 
        
        # sidebar sliders
        size = st.slider('Total floor area (ft2)', 100, 1000,
                                value=400, 
                                help='Select the total square footage of floor (maximum floor area per Seattle code is 1000ft2)',
                                step=10, key='ft2')
        wwr = st.slider('Window-to-wall ratio', 0.0, 0.9, 
                                help='Window to wall ratio is the ratio between glazing (window) surface area and opaque surface area',
                                value=.4, key='wwr')
        polyiso_t = st.slider('Polyiso insulation depth (inches)', 0.0, 1.0, 
                                      help='Select amount of polyiso insulation in wall assembly',
                                      step=.25, value=.75, key='polyiso')
        cellulose_t = st.slider('Cellulose insulation depth (inches)', 0.0, 10.0, 
                                        help='Select amount of cellulose insulation in wall assembly',
                                        step=.5, value=8.0, key='cellulose')
            
        site = sited[site]

        num_stories = typologyd[typology]['num_stories']
        num_units = typologyd[typology]['num_units']
        if num_stories == 1:
            height = 10
            footprint = size
        else:
            height = 20
            footprint = size / 2.

        assembly_r = round(float(calc_r(polyiso_t, cellulose_t)), 2)
        
        length = sqrt(footprint)
        volume = (length ** 2) * height 
        surf_area = ((length ** 2) * 2) + (4 * (length * height))
        surf_vol_ratio = surf_area / volume
        
        # show r-assembly value
        st.text('R-assembly: ' + str("%.2f" % assembly_r) + '(ft2·°F·h/BTU)')
        
        # submit user prediction
        pred_1, pred_2, pred3 = st.columns([1,1,1])
        with pred_2:
            activate = st.form_submit_button(label='Predict', 
                            help='Click \"Predict\" once you have selected your desired options')
        # if st.button('Favorite', help=
        #     'Add to list of favorite combinations to easily return to result'):
        # csv_favs = convert_df(user_favorites(results, count))
        # pass #TODO
    
    with st.sidebar:    
        now = datetime.datetime.now()
        file_name_all = 'results_' + (now.strftime('%Y-%m-%d_%H_%M')) + '.csv'
        csv_all = convert_df(st.session_state.results)
        
        with st.expander('Advanced'):
            duration = st.select_slider(
                label='Prediction period',
                options=['1 month',
                        '6 months',
                        '1 year',
                        '5 years',
                        '10 years'],
                value='1 year',
                )
            duration_num = durationd[duration]
            advanced_toggle = st.checkbox('Show dataframe',
                                    help='Use scrollbar to view additional columns if your display does not support viewing all')
        
        side_col1, side_col2, side_col3 = st.columns([5,1,4])
        with side_col1:
            st.download_button('Download results',
                            data=csv_all, file_name=file_name_all,
                            help='Download a .CSV spreadsheet with all simulation data from current session')
        with side_col3:
        # clear results
            clear_res = st.button('Clear results',
                                help='Delete all previous prediction data from current session')
            
    if activate:
        count = len(st.session_state.results.index) + 1
            
        pred_input = create_input_df(site, size, footprint, height, num_stories, num_units, inf_rate, orientation, wwr,
                                setback, assembly_r, surf_vol_ratio)

        eui = predict_eui(pred_input, model) / mshp_cop
        
        # convert kBTU/ft2 to kWh/m2
        eui_kwh = eui * 3.2 
        # convert kBTU/ft2 to kWh, then multiply by CO2 equivalent of grid (Seattle)
        co2 = eui_kwh * size * 0.09290304 * kgCO2e 
        # convert kBTU/ft2 to kWh, then multiply by average cost per kWh (Seattle)
        cost = eui_kwh * size * 0.09290304 * kwh_cost
        
        rounded_eui = round(float(eui), 2)
        rounded_eui_kwh = round(float(eui_kwh), 2)
        rounded_co2 = round(float(co2), 2)
        rounded_cost = round(float(cost), 2)
        
        outcomes_dict = {
            'site': site,
            'size': size,
            'footprint': footprint,
            'height': height,
            'num_stories': num_stories,
            'num_units': num_units,
            'inf_rate': inf_rate,
            'orientation': orientation,
            'wwr': wwr,
            'setback': setback,
            'assembly_r': assembly_r,
            'surf_vol_ratio': surf_vol_ratio,
            'eui_kwh': rounded_eui_kwh,
            'eui_kbtu': rounded_eui,
            'annual_carbon': rounded_co2,
            'annual_cost': rounded_cost
        }
        outcomes = pd.DataFrame(outcomes_dict, index=[0])
        st.session_state.results = st.session_state.results.append(outcomes, ignore_index=True)
        
    with col1:
        
        if count == 0:
            st.metric('Predicted EUI (energy use intensity)', None)
            st.write('\n' + '\n')
            st.metric('Predicted operational carbon (' + duration + ')', None)
            st.write('\n' + '\n')
            st.metric('Predicted  energy cost (' + duration + ')', None) 
            st.write('\n' + '\n')
            
        if count == 1:
            display_co2 = round(float(rounded_co2 * duration_num), 2 )
            display_cost = round(float(rounded_cost * duration_num), 2)
            st.metric('Predicted EUI (energy use intensity)', str(rounded_eui) + ' kBTU/ft2')
            st.metric('Predicted operational carbon (' + duration + ')', str(display_co2) + ' kgCO2')
            st.metric('Predicted  energy cost (' + duration + ')', '$' + str(display_cost))  
            
        if count > 1:
            display_co2 = round(float(rounded_co2 * duration_num), 2 )
            display_cost = round(float(rounded_cost * duration_num), 2)
            d_eui_kbtu = percent_change(
                st.session_state.results.iat[count - 2, st.session_state.results.columns.get_loc('eui_kbtu')], 
                rounded_eui)
            d_carbon = percent_change(
                round(float(st.session_state.results.iat[count - 2, st.session_state.results.columns.get_loc('annual_carbon')] * duration_num), 2), 
                display_co2)
            d_cost = percent_change(
                round(float(st.session_state.results.iat[count - 2, st.session_state.results.columns.get_loc('annual_cost')] * duration_num), 2), 
                display_cost)
            st.metric('Predicted EUI (energy use intensity)', ("%.2f" % rounded_eui) + ' kBTU/ft2', delta=("%.1f" % d_eui_kbtu) + ' %', delta_color='inverse')
            st.metric('Predicted operational carbon (' + duration + ')', ("%.2f" % display_co2) + ' kgCO2', delta=("%.1f" % d_carbon) + ' %', delta_color='inverse')
            st.metric('Predicted  energy cost (' + duration + ')', '$' + ("%.2f" % display_cost), delta=("%.1f" % d_cost) + ' %', delta_color='inverse')  
        
    # model viewer    
    with col2:    
        mesh = make_mesh.make_mesh(size, wwr, num_stories, num_units)
        st.plotly_chart(mesh, use_container_width=True)
    
    with st.container():
        # st.subheader('Plot options:')
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1: 
            x_axis_data = st.selectbox('X-axis', options=['Lot type', 'Infiltration rate', 'Orientation',
                                    'Setbacks', 'Floor area', 'WWR', 'R-assembly'], index=6, help='Select data feature to display on X axis')   
        with s_col2:
            y_axis_data = st.selectbox('Y-axis', options=['EUI', 'CO2', 'Cost'], index=0, help='Select data feature to display on Y axis')
        
        with s_col3:
            colorby = st.selectbox('Color by', options=['Lot type', 'Infiltration rate', 'Orientation',
                                    'Setbacks', 'Floor area', 'WWR', 'R-assembly'], help='Select data feature to color markers by')                    
            
        if count > 0:
            
            fig = plot_scatter(
                st.session_state.results[plotd[x_axis_data]], 
                st.session_state.results[plotd[y_axis_data]], 
                st.session_state.results[plotd[colorby]],
                x_axis_data,
                y_axis_data
                )
            st.plotly_chart(fig, use_container_width=True)
            
    if clear_res:
        st.session_state.results = st.session_state.results[0:0]

    if advanced_toggle:
        st.dataframe(st.session_state.results)
        
    with st.expander('Documentation'):
        st.markdown('How to use: \n')
        st.markdown('1. Select design parameter values in the left sidebar \n')
        st.markdown('2. Choose "Predict" to view results and visualize simple model \n')
        st.markdown('3. Compare results using scatter plot below \n')
        st.markdown('4. Click "Download results" to download a spreadsheet containing all inputs and results \n \n')
        st.markdown('Note: energy and kgCO2 values in downloadable spreadsheet are *annual* \n \n')
        st.markdown('Questions or feedback? Open an \'issue\' here https://github.com/prxsto/dadu-predictor')
            
st.set_page_config(layout='wide')


if __name__=='__main__':
    for i in range(50): print('')
    model = load_model()
    web_tool(model)

# to run: streamlit run /Users/prxsto/Documents/GitHub/msdc-thesis/web_tool.py 