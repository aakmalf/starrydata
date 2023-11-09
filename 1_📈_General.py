import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import kde
from scipy.interpolate import griddata
import os

from functools import lru_cache


import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

st.set_page_config(layout="wide", page_title="StarryData Visualization", page_icon="📈")

@st.cache_data
def load_data():
    df = pd.read_csv('thermo.csv')
    return df

df = load_data()

def plot_value_counts_material_family(series, column_name, top_n=10):
    value_counts = series.value_counts().sort_values(ascending=False)
    
    if top_n is not None:
        value_counts = value_counts.nlargest(top_n)
        
    value_counts = value_counts[::-1]
    
    fig, ax = plt.subplots()
    
    bars = ax.barh(value_counts.index, value_counts.values, linewidth=1.5)  
    
    for i, v in enumerate(value_counts.values):
        ax.text(v + 0.1, i, str(v), color='black', va='center') 
    
    plt.xlabel('Counts', fontweight='bold')
    plt.ylabel(column_name, fontweight='bold')
    plt.title(f'Value Counts Bar Plot of {column_name}', fontweight='bold')
    
    ax.set_xlim(0, max(value_counts.values) * 1.1)
    
    st.pyplot(fig)

def plot_value_counts_properties(series, column_name, top_n=10):
    value_counts = series.value_counts().sort_values(ascending=False)
    
    if top_n is not None:
        value_counts = value_counts.nlargest(top_n)
        
    value_counts = value_counts[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 9.8))
    
    bars = ax.barh(value_counts.index, value_counts.values, linewidth=1.5, color='green')
    
    for i, v in enumerate(value_counts.values):
        ax.text(v + 0.1, i, str(v), color='black', va='center') 
    
    plt.xlabel('Counts', fontweight='bold')
    plt.ylabel(column_name, fontweight='bold')
    plt.title(f'Value Counts Bar Plot of {column_name}', fontweight='bold')
    
    ax.set_xlim(0, max(value_counts.values) * 1.1)
    
    st.pyplot(fig)




### Streamlit App ###

title = st.container()
information = st.container()
general_plot = st.container()
dataframe = st.container()
scatterplot = st.container()
sidebar = st.sidebar

# boxplot = st.container()
# temperature = st.container()


with sidebar :
    st.sidebar.header('User Input Features')
    family_filter = st.sidebar.multiselect('Select material family', df['materialfamily'].unique())
    properties_filter = st.sidebar.multiselect('Select properties', df['propertyname_y'].unique())

    if properties_filter:  # Check if there is at least one property selected
        st.sidebar.markdown('##### Enter ranges for selected properties:')

    # Dictionary to hold the user input values for min and max for each property
    user_inputs = {}

    # If the user has selected at least one property, show input fields for min and max
    for property_name in properties_filter:
        # Create a subheader for the property using markdown for better visual distinction
        st.sidebar.markdown(f"**{property_name}**")
        
        # For each property, create an indented section with a left arrow symbol
        left, right = st.sidebar.columns((1, 20))
        left.write("↳")
        
        with right:
            min_val = st.text_input(f"Minimum value for {property_name}", key=f"min_{property_name}")
            max_val = st.text_input(f"Maximum value for {property_name}", key=f"max_{property_name}")

        user_inputs[property_name] = {'min': min_val, 'max': max_val}


# ... (previous code)

# Initialize filter_df with the full DataFrame
filter_df = df

# Apply material family filter if selected
if family_filter:
    filter_df = filter_df[filter_df["materialfamily"].isin(family_filter)]

# Apply properties filter if any properties are selected
if properties_filter:
    # Filter by selected properties
    selected_properties_df = filter_df[filter_df["propertyname_y"].isin(properties_filter)]
    
    # Initialize an empty DataFrame for filtered results
    filtered_properties_df = pd.DataFrame(columns=selected_properties_df.columns)

    # For each selected property, apply additional filters based on user input
    for property_name in properties_filter:
        # Select rows for current property only
        current_property_df = selected_properties_df[selected_properties_df["propertyname_y"] == property_name]
        
        # Ensure user_inputs for the property exist
        if property_name in user_inputs:
            # Retrieve user input values
            min_val = user_inputs[property_name].get('min')
            max_val = user_inputs[property_name].get('max')

            # Apply min filter if a min value is provided and is a valid number
            if min_val :
                current_property_df = current_property_df[current_property_df["y_value_at_300"] >= float(min_val)]

            # Apply max filter if a max value is provided and is a valid number
            if max_val :
                current_property_df = current_property_df[current_property_df["y_value_at_300"] <= float(max_val)]

        # Combine the current property's filtered results back into the main filtered DataFrame
        filtered_properties_df = pd.concat([filtered_properties_df, current_property_df])

    # If filtered_properties_df is not empty, update filter_df
    if not filtered_properties_df.empty:
        filter_df = filtered_properties_df

number_of_data = filter_df.shape[0]
number_of_sample = filter_df["sampleid"].nunique()

with title :
    st.title('StarryData Visualization TE Experiment')
    # if family_filter:
    #     st.markdown(f"#### Material Family: {family_filter[0]}")
    # else:
    #     st.markdown("#### Material Family: All")
    st.markdown(" ---")


with information :
    col1, col2 = st.columns(2)

    with col1 :
        st.metric(label=f"##### Total Data", value=number_of_data)
    
    with col2 :
        st.metric(label=f"##### Total Sample", value=number_of_sample)

    st.markdown(" ---")

with general_plot :
    st.subheader('General Plot')
    col1, col2 = st.columns(2)
    with col1:
        # fig, ax = plt.subplots()
        plot_value_counts_material_family(filter_df['materialfamily'], 'materialfamily' )

    with col2:
        plot_value_counts_properties(filter_df['propertyname_y'], 'properties' )

    st.markdown("---",)

with dataframe :
    
    @lru_cache(maxsize=None)
    def calculate_pivot(data, **kwargs):
        """u
        Calculates a pivot table for a given material and data.

        Parameters:
        data (pandas.DataFrame): The data to calculate the pivot table from.
        **kwargs: Variable number of keyword arguments, where each key-value pair represents the property name and the associated value.

        Returns:
        pandas.DataFrame: The pivot table.
        """
        properties = list(kwargs.values())
        material_data = data[data["propertyname_y"].isin(properties)]
        pivot = material_data.pivot_table(values='y_value_at_300', index='sampleid', columns='propertyname_y', aggfunc='mean').reset_index()
        pivot_final = pivot.merge(material_data[['sampleid', 'materialfamily', 'composition']], on='sampleid', how='left')
        pivot_final = pivot_final.drop_duplicates()
        pivot_final = pivot_final.dropna(subset=properties)
        pivot_final.loc[pivot_final["materialfamily"] == "", "materialfamily"] = "Unknown"
        pivot_final.loc[pd.isna(pivot_final["materialfamily"]), "materialfamily"] = "Unknown"
        
        # You can calculate 'result' for multiple properties
        #pivot_final[f"result"] = pivot_final[properties[1]]**2/pivot_final[properties[0]]

        return pivot_final
    
   

    pivot_final = calculate_pivot(filter_df, **{f"property_{i}": prop for i, prop in enumerate(properties_filter)})
    # Now you call the function with the selected properties
    # pivot_final = calculate_pivot(df, properties_filter)

    # Do something with pivot_table, such as displaying it or further processing

    
    st.markdown("#### Pivot table material properties at 300K")
    st.markdown(f"Total data or sample plot : {pivot_final.shape[0]}")
    
    page_size = 5

    # Use session state to store the current start index of the displayed page
    if 'start_idx' not in st.session_state:
        st.session_state.start_idx = 0  # Initialize start index

    format_mapping = {
        # 'Seebeck coefficient': '{:.6f}',  # Format as a floating-point number with 2 decimal places
        'Electrical resistivity': '{:.2e}',  # Format in scientific notation
        # Add more columns and formats as needed
    }

    # Check if properties_filter is not empty, and if so, use it to update format_column
    if properties_filter:
        format_column = properties_filter

    # Apply formatting to the specified columns in format_mapping
    for column, format_str in format_mapping.items():
        if column in properties_filter:
            pivot_final[column] = pivot_final[column].apply(lambda x: format_str.format(x))
    # Display the current page of the DataFrame
    st.dataframe(pivot_final.iloc[st.session_state.start_idx:st.session_state.start_idx + page_size])

    # Pagination buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Previous'):
            # Go to the previous page
            if st.session_state.start_idx > 0:
                st.session_state.start_idx -= page_size
    with col2:
        if st.button('Next'):
            # Go to the next page
            if st.session_state.start_idx + page_size < len(df):
                st.session_state.start_idx += page_size

    # Use a placeholder to clear the previous dataframe from view
    placeholder = st.empty()

    st.markdown(" ---")

if len(properties_filter) > 1 and "ZT" in properties_filter:

    with scatterplot :


        def scatterplot_thermal_pf(data, x_feature="Thermal conductivity", y_feature="Power factor", on=False, hue = None):
            
            num_data = data.shape[0]  # Get the number of data

            plt.figure(figsize=(10, 6))


            # Create the scatter plot
            sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=hue, palette='bright', alpha=0.7, s=50)

            # Annotate the top 5 materials with an offset if annotate is True
            top5_materials = data.sort_values(by="ZT", ascending=False).head(5)
            if on :
                offset = (5, 5)  # You can adjrust the offset as needed
                for index, row in top5_materials.iterrows():
                    plt.annotate(
                        f"{row['composition']} ({row['materialfamily']})",
                        (row[x_feature], row[y_feature]),
                        textcoords="offset points",
                        xytext=offset
                    )

            unit_x = df[df["propertyname_y"]==x_feature]["unitname_y"].value_counts().index[0]
            unit_y = df[df["propertyname_y"]==y_feature]["unitname_y"].value_counts().index[0]

            plt.title(f"{x_feature} vs {y_feature}", fontweight='bold')
            plt.xlabel(f"{x_feature} ({unit_x})", fontweight='bold')
            plt.ylabel(f"{y_feature} ({unit_y})", fontweight='bold')
        
            # After creating the original plot (e.g., in a Jupyter notebook)
            
        
            plt.legend(title=f"Number of data: {num_data}", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            if x_feature == "Electrical resistivity":
                plt.xscale("log")
                plt.xlabel(f"Log {x_feature} ({unit_x})")
            if y_feature == "Electrical resistivity":
                plt.yscale("log")
                plt.ylabel(f"Log {y_feature} ({unit_y})")

            plt.tight_layout()
            st.pyplot(plt)

        def scatterplot_thermal_pf_heatmap(data,x_feature="Thermal conductivity", y_feature="Power factor", on=False):
            plt.figure(figsize=(10, 6))

            top5_materials = data.sort_values(by="ZT", ascending=False).head(5)

            # Create a colormap
            colormap = plt.cm.magma

            # Create a Normalize object
            norm = mcolors.Normalize(vmin=data['ZT'].min(), vmax=data['ZT'].max())

            # Create the scatter plot with a color bar
            scatter = plt.scatter(data[x_feature], data[y_feature], c=data['ZT'], cmap=colormap, norm=norm, alpha=0.7, s=50)

            # Annotate the top 5 materials with an offset if annotate is True
            if on:
                offset = (5, 5)  # You can adjust the offset as needed
                for index, row in top5_materials.iterrows():
                    plt.annotate(
                        f"{row['composition']} ({row['materialfamily']})",
                        (row[x_feature], row[y_feature]),
                        textcoords="offset points",
                        xytext=offset
                    )

            unit_x = df[df["propertyname_y"]==x_feature]["unitname_y"].value_counts().index[0]
            unit_y = df[df["propertyname_y"]==y_feature]["unitname_y"].value_counts().index[0]

            plt.title(f"{x_feature} vs {y_feature}")
            plt.xlabel(f"{x_feature} ({unit_x})")
            plt.ylabel(f"{y_feature} ({unit_y})")

            if x_feature == "Electrical resistivity":
                plt.xscale("log")
                plt.xlabel(f"Log {x_feature} ({unit_x})")
            if y_feature == "Electrical resistivity":
                plt.yscale("log")
                plt.ylabel(f"Log {y_feature} ({unit_y})")
      

            # Add a color bar
            cbar = plt.colorbar(scatter)
            cbar.set_label("ZT")

            plt.tight_layout()
            st.pyplot(plt)
                       

        def scatterplot_thermal_pf_heatmap_with_contour(data: pd.DataFrame, 
                                                        x_feature: str = "Thermal conductivity", 
                                                        y_feature: str = "Power factor", 
                                                        on: bool = True) -> None:
            """Generate a scatterplot with heatmap coloring and contour lines based on point density."""

            plt.figure(figsize=(10, 6))

            # Getting the top 5 materials based on ZT
            top5_materials = data.sort_values(by="ZT", ascending=False).head(5)

            # Setting up the colormap and normalization
            colormap = plt.cm.magma
            norm = mcolors.Normalize(vmin=data['ZT'].min(), vmax=data['ZT'].max())

            # Convert to numeric and handle non-numeric entries
            x = pd.to_numeric(data[x_feature], errors='coerce')
            y = pd.to_numeric(data[y_feature], errors='coerce')

            # Drop NaN values that result from the conversion
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            # Perform the kernel density estimate
            nbins = 100
            k = kde.gaussian_kde([x,y])
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            # Plotting the contour
            contour = plt.contourf(xi, yi, zi.reshape(xi.shape), cmap="viridis", alpha=0.7)

            # Plotting the scatter plot over the contour
            scatter = plt.scatter(x, y, c=data.loc[mask, 'ZT'], cmap=colormap, norm=norm, alpha=0.7, s=20)

            # Annotation for the top 5 materials
            if on:
                offset = (5, 5)
                for _, row in top5_materials.iterrows():
                    plt.annotate(
                        f"{row['composition']} ({row['materialfamily']})", 
                        (row[x_feature], row[y_feature]), 
                        textcoords="offset points", 
                        xytext=offset
                    )

            # Assuming 'df' is the DataFrame that contains the units for the features
            unit_x = df[df["propertyname_y"]==x_feature]["unitname_y"].value_counts().index[0]
            unit_y = df[df["propertyname_y"]==y_feature]["unitname_y"].value_counts().index[0]

            plt.xlabel(f"{x_feature} ({unit_x})")
            plt.ylabel(f"{y_feature} ({unit_y})")

            # Adding title and colorbar
            plt.title(f"{x_feature} vs {y_feature}")
            cbar = plt.colorbar(contour, format='')
            cbar.set_label("Density Data")

            # Display the plot
            plt.tight_layout()
            st.pyplot(plt)  # Use plt.show() in a local environment. Replace with st.pyplot() for Streamlit.

        # To use this function in a Streamlit app, make sure to pass a properly formatted DataFrame
        # and uncomment the st.pyplot() call at the end of the function.







        st.markdown(f"#### Scatterplot of {properties_filter[0]} vs {properties_filter[1]} at 300K")
        st.write(f"Total data or sample plot : {pivot_final.shape[0]}") 

        cola, colb, colc = st.columns(3)

        with cola:
            hue = st.toggle(f"Color by material family", False)

        with colb:
            on = st.toggle(f"Annotate top 5 ZT materials", False)
        
        with colc:
            countour = st.toggle("Add Density Data Color", False)
        
        col1, col2 = st.columns(2)


        with col1:
            st.markdown("\n\n")

            if hue:
                scatterplot_thermal_pf(pivot_final, x_feature=properties_filter[0], y_feature=properties_filter[1], hue="materialfamily", on = on)
            else:
                scatterplot_thermal_pf(pivot_final, x_feature=properties_filter[0], y_feature=properties_filter[1], on = on)


        with col2:
            st.markdown("\n\n")

            if countour:
                scatterplot_thermal_pf_heatmap_with_contour(pivot_final, x_feature=properties_filter[0], y_feature=properties_filter[1], on=on)
            else:
                scatterplot_thermal_pf_heatmap(pivot_final, x_feature=properties_filter[0], y_feature=properties_filter[1], on=on)

        st.markdown(" ---")


if "Carrier concentration" not in properties_filter and len(properties_filter) > 1 and "ZT" in properties_filter:

    # area_color = st.toggle("Show contour plot", False)

    with st.expander(f"##### Contour plot of {properties_filter[0]} vs {properties_filter[1]} vs ZT") :
        
        # if area_color:
        # st.markdown(f"#### contour plot {properties_filter[0]} vs {properties_filter[1]} vs ZT")

        

        

        def scatterplot_thermal_pf_heatmap_with_contour(data: pd.DataFrame, 
                                                    x_feature: str, 
                                                    y_feature: str, 
                                                    z_feature: str = "ZT", 
                                                    on: bool = True) -> None:
            """Generate a scatterplot with heatmap coloring and contour lines based on ZT values."""

            plt.figure(figsize=(10, 6))

            colormap = plt.cm.magma
            norm = mcolors.Normalize(vmin=data['ZT'].min(), vmax=data['ZT'].max())

            # Convert to numeric and handle non-numeric entries
            x = pd.to_numeric(data[x_feature], errors='coerce')
            y = pd.to_numeric(data[y_feature], errors='coerce')
            z = pd.to_numeric(data[z_feature], errors='coerce')

            # Drop NaN or infinite values that result from the conversion
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x = x[mask]
            y = y[mask]
            z = z[mask]

            # Check that we have enough data points after removing NaNs and Infs
            if x.size == 0 or y.size == 0 or z.size == 0:
                raise ValueError("After cleaning the data, no valid points remain.")

            # Create grid coordinates for contour plot
            xi = np.linspace(x.min(), x.max(), 20)
            yi = np.linspace(y.min(), y.max(), 20)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate z values on a grid defined by X and Y
            zi = griddata((x, y), z, (xi, yi), method='linear')

            # Plot contour lines
            contour_lines = plt.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
            contour_filled = plt.contourf(xi, yi, zi, levels=14, cmap=colormap)

            # Plotting the scatter plot over the contour
            scatter = plt.scatter(x, y, c=z, cmap=colormap, s=30)

            # Adding colorbar
            cbar = plt.colorbar(contour_filled)
            cbar.set_label(z_feature)

            # Adding titles and labels
            unit_x = df[df["propertyname_y"]==x_feature]["unitname_y"].value_counts().index[0]
            unit_y = df[df["propertyname_y"]==y_feature]["unitname_y"].value_counts().index[0]

            plt.xlabel(f"{x_feature} ({unit_x})")
            plt.ylabel(f"{y_feature} ({unit_y})")

            # Adding title and colorbar
            plt.title(f'Scatter plot with contour lines for {x_feature} vs {y_feature}')

            st.pyplot(plt)


        # if area_color:

        scatterplot_thermal_pf_heatmap_with_contour(pivot_final, x_feature=properties_filter[0], y_feature=properties_filter[1], on=on)
        
        st.markdown(" ---")



if len(properties_filter)>0:

    with st.expander(f"##### Show boxplot of selected properties"):

        # st.markdown(f" Boxplot of {properties_filter[0]} and {properties_filter[1]}")

        fig, ax = plt.subplots(nrows=len(properties_filter), ncols=1, figsize=(8, 6))

        # Define a list of colors
        colors = ['red', 'green', 'blue', 'orange', 'purple']

        for i, pro in enumerate(properties_filter):
            # Pass the color to the palette parameter
            sns.boxplot(x=pro, data=pivot_final, ax=ax[i], palette=[colors[i]])
            plt.tight_layout()
        
        st.pyplot(fig)      

        st.markdown(" ---")

### Temperature Dependence ###




    
if len(properties_filter) > 0:

    with st.expander(f"##### Temperature dependence"):

        def data_temp(df=filter_df, propertyname="Seebeck coefficient", min_temp=300, max_temp=1000,hue="ZT", min_data = -10e50, max_data = 10e50):
            """
            Filter data based on column and value
            """
            data = df[df["propertyname_y"].isin([propertyname, "ZT"])]
            pivot_data = data.pivot_table(index="sampleid", columns="propertyname_y", values="data", aggfunc="first").reset_index()
            pivot_data.dropna(inplace=True)
            pivot_data.drop_duplicates(subset="sampleid", inplace=True)
            pivot_final = pivot_data.merge(df[['sampleid', 'materialfamily', 'composition']], on='sampleid', how='left')
            pivot_final.loc[pivot_final["materialfamily"] == "", "materialfamily"] = "Unknown"
            pivot_final.loc[pd.isna(pivot_final["materialfamily"]), "materialfamily"] = "Unknown"

            all_x_values = []
            all_y_values = []
            hue_values = []

            # Helper function to handle iterable and non-iterable values
            # def get_iterable(value):
            #     if isinstance(value, (list, tuple)):
            #         return value
            #     elif isinstance(value, dict):
            #         return [value]
            #     else:
            #         return [value]

            import ast

    # Helper function to handle iterable and non-iterable values
            def get_iterable(value):
                if isinstance(value, str):
                    try:
                        # Safely evaluate string as a Python literal
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        # Handle the case where the string is not a valid Python literal
                        # This depends on your data; you might want to return an empty list or dictionary
                        # or handle it some other way
                        
                        value = []

                if isinstance(value, (list, tuple)):
                    return value
                elif isinstance(value, dict):
                    return [value]
                else:
                    return [value]


            # Iterate through each row in the dataset
            for _, row in pivot_final.iterrows():
                
                prop = get_iterable(row[property_name])
                zt_values = get_iterable(row['ZT'])
                
                for point_s, point_z in zip(prop, zt_values):
                    
                    # print(f"Type of point_s: {type(point_s)}, Value: {point_s}")  # Add this line
                    # print(f"Type of point_z: {type(point_z)}, Value: {point_z}")  # Add this line
                    
                    if min_temp <= point_s['x'] < max_temp and min_data < point_s['y'] < max_data:
                        all_x_values.append(point_s['x'])
                        all_y_values.append(point_s['y'])

                        if hue == "ZT":
                            hue_values.append(point_z['y'])
                        elif hue == "materialfamily":
                            hue_values.append(row['materialfamily'])

            # x_filtered = [x for x in all_x_values if x >= 300 and x <= 1000 ]
            # y_filtered = [y for y in all_y_values if y >= -10 and y <= 0.005 ]
            # z_filtered = [z for z in hue_values if z >= 0 and z <= 2.5 ]

            # return x_filtered, y_filtered, z_filtered
            return all_x_values, all_y_values, hue_values
        

            
        def scatter_plot_prop_temp_safe(all_x_values, all_y_values, hue_values, property= "Seebeck coefficient", hue="ZT", log=False, annotate=False):

            fig, ax = plt.subplots(figsize=(8, 4))

            # Plotting
            if hue == "ZT":
                vmin = min(hue_values)
                vmax = max(hue_values)

                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                colormap = plt.cm.magma

                scatter = sns.scatterplot(x=all_x_values, y=all_y_values, c=hue_values, cmap=colormap, norm=norm, alpha=0.7, s=10, palette='viridis', legend=True, ax = ax)
                scatter = ax.collections[0]


                cbar = plt.colorbar(scatter)
                cbar.set_label("ZT")
            else:
                sns.scatterplot(x=all_x_values, y=all_y_values, hue=hue_values, s=10, ax = ax)
                scatter = ax.collections[0]
                plt.legend(title='Material family', bbox_to_anchor=(1.05, 1), loc="upper left")



            unit_y = df[df["propertyname_y"]==property]["unitname_y"].value_counts().index[0]

            
            plt.xlabel(f"Temperature (K)")
            plt.ylabel(f"{property} ({unit_y})")

            # Adding title and colorbar
            plt.title(f"{property} vs Temperature")

            # Add a color bar


            plt.tight_layout()
            st.pyplot(fig)
            # Load the dataset
            # dataset = pd.read_csv('/mnt/data/dataset.csv')
            # Convert string representation of lists to actual lists
            # dataset['Seebeck coefficient'] = dataset['Seebeck coefficient'].apply(ast.literal_eval)
            # dataset['ZT'] = dataset['ZT'].apply(ast.literal_eval)

            # Test the function with this dataset


        left, right = st.columns(2)

        with left:
            min_temperature = st.number_input("Minumum Temperature", value=300, placeholder="Type a number... (Default 300))", step = 10)
    
        
        with right:
            max_temperature = st.number_input("Maximum Temperature", value=1000, placeholder="Type a number...(Default 1000)", step = 10)


        hue = st.radio("Color by", ("ZT", "materialfamily"))

        

        tab = st.tabs(properties_filter)

        
        for i,property_name in enumerate(properties_filter):
            if property_name != "ZT":
                
            # st.write(f"##### {property_name}")


                with tab[i] :
                    left, right = st.columns(2)

                    with left:
                        min_data = st.number_input(f"Minumum  {property_name}", placeholder="Type a number... ", step = 0.1, value = -1*10e50)
                
                    
                    with right:
                        max_data = st.number_input(f"Maximum {property_name}", placeholder="Type a number... ", step = 0.1, value = 10e50)
                # with left:

                    x,y,z = data_temp(df=filter_df, 
                                        propertyname=property_name, 
                                        min_temp=min_temperature, 
                                        max_temp=max_temperature,
                                        min_data = min_data,
                                        max_data= max_data,
                                        hue=hue)

                    scatter_plot_prop_temp_safe(x,y,z, property=property_name, hue=hue, log=False, annotate=False)

                    st.markdown(" ***")        

            else:

                with tab[i]:
                    
                    left, right = st.columns(2)

                    with left:
                        min_data = st.number_input(f"Minumum  {property_name}", placeholder="Type a number... (Default 300))", step = 0.1, value = -1*10e50)
                
                    
                    with right:
                        
                        max_data = st.number_input(f"Maximum {property_name}", placeholder="Type a number...(Default 1000)", step = 0.1, value = 10e50)

                    
                    x,y,z = data_temp(df=filter_df, propertyname="ZT", min_temp=300, max_temp=1000,hue="materialfamily", min_data=min_data, max_data=max_data)

                    scatter_plot_prop_temp_safe(x,y,z, property="ZT", hue="materialfamily", log=False, annotate=False)
            

    #     fig, ax = plt.subplots()

    # # Create a scatter plot on the axes
    #     sns.scatterplot(x=x, y=y, ax=ax)

    #     # Display the figure using Streamlit
    #     st.pyplot(fig)

