import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import kde
from scipy.interpolate import griddata
import os
import ast

import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

st.set_page_config(layout="wide", page_title="Material Category", page_icon="📊")


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
category = st.container()
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

    if properties_filter:  
        st.sidebar.markdown('##### Enter ranges for selected properties:')

    user_inputs = {}

    for property_name in properties_filter:
        st.sidebar.markdown(f"**{property_name}**")
        
        left, right = st.sidebar.columns((1, 20))
        left.write("↳")
        
        with right:
            min_val = st.text_input(f"Minimum value for {property_name}", key=f"min_{property_name}")
            max_val = st.text_input(f"Maximum value for {property_name}", key=f"max_{property_name}")

        user_inputs[property_name] = {'min': min_val, 'max': max_val}


filter_df = df

if family_filter:
    filter_df = filter_df[filter_df["materialfamily"].isin(family_filter)]

if properties_filter:
    selected_properties_df = filter_df[filter_df["propertyname_y"].isin(properties_filter)]
    
    filtered_properties_df = pd.DataFrame(columns=selected_properties_df.columns)

    for property_name in properties_filter:
        current_property_df = selected_properties_df[selected_properties_df["propertyname_y"] == property_name]
        
        if property_name in user_inputs:
            min_val = user_inputs[property_name].get('min')
            max_val = user_inputs[property_name].get('max')

            if min_val :
                current_property_df = current_property_df[current_property_df["y_value_at_300"] >= float(min_val)]

            if max_val :
                current_property_df = current_property_df[current_property_df["y_value_at_300"] <= float(max_val)]

        filtered_properties_df = pd.concat([filtered_properties_df, current_property_df])

    if not filtered_properties_df.empty:
        filter_df = filtered_properties_df

with title :
    st.title('Categorize of Material StarryData')
    # if family_filter:
    #     st.markdown(f"#### Material Family: {family_filter[0]}")
    # else:
    #     st.markdown("#### Material Family: All")
    st.markdown(" ---")



with category :

    st.write(f"#### Set Up Material Category")
    st.write("set up material category based on resistivity value at 300K")

    left, mid, right = st.columns(3)
    
    with left :
        st.write(f"###### Insulator")

        min_insu = st.number_input("Min Resistivity (ohm.m) ", value=1e5, format="%.2e", placeholder="Type a number... )")
        max_insu = st.number_input("Max Resistivity (ohm.m)  ", value=1e16, format="%.2e", placeholder="Type a number... )")

    with mid :
        st.write(f"###### Semiconductor")

        min_semi = st.number_input("Min Resistivity (ohm.m) ", value=1e-3, format="%.2e", placeholder="Type a number... )")
        max_semi = st.number_input("Max Resistivity (ohm.m)  ", value=1, format="%.2e", placeholder="Type a number... )")

    with right :
        st.write(f"###### Metal")

        min_metal = st.number_input("Min Resistivity (ohm.m) ", value=1e-8, format="%.2e", placeholder="Type a number... )")
        max_metal = st.number_input("Max Resistivity (ohm.m)  ", value=1e-6, format="%.2e", placeholder="Type a number... )")

    st.markdown(" ---")
   
    def category(x):
        if min_metal <= x <= max_metal:
            return "Metal"
        elif min_semi <= x <= max_semi:
            return "Semiconductor"
        elif min_insu <= x <= max_insu:
            return "Insulator"
    
    if len(properties_filter) > 0:
        list_sample = filter_df["sampleid"].unique()
        filter_df = df[df["sampleid"].isin(list_sample)]


    resis = filter_df[filter_df["propertyname_y"] == "Electrical resistivity"]
        
    resis["category"] = resis["y_value_at_300"].apply(category)



    import ast

    def get_iterable(value):
        if isinstance(value, str):
            try:

                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                
                value = []

        if isinstance(value, (list, tuple)):
            return value
        elif isinstance(value, dict):
            return [value]
        else:
            return [value]
    
    
    def create_scatter_plot_log_color(df, column='category'):

        plt.figure(figsize=(7, 4.7))
        
        categories = df[column].unique()
        colors = {
            'Metal': 'red',
            'Semiconductor': 'green',
            'Insulator': 'blue'
        }

        for i, category in enumerate(categories):
            category_data = df[df[column] == category]
            all_x_values = []
            all_y_values = []


            for _, row in category_data.iterrows():

                data = get_iterable(row['data'])

                x_values = [point['x'] for point in data if 300 <= point['x'] < 1000]
                y_values = [point['y'] for point in data if 300 <= point['x'] < 1000]
                all_x_values.extend(x_values)
                all_y_values.extend(y_values)

            plt.scatter(all_x_values, all_y_values, color=colors.get(category, 'orange'), label=category)

        plt.xlabel('Temperature (K)', fontweight='bold')
        plt.ylabel('Log Resistivity (ohm.m))', fontweight='bold')
        plt.title('Scatter Plot Temperature vs Resistivity', fontweight='bold')
        plt.yscale('log')  
        plt.legend()
        st.pyplot(plt)

    a, b = st.columns(2)

    with a :
        create_scatter_plot_log_color(resis, column='category')

    with b:
        def plot_value_counts_material_category(series, column_name, top_n=10):

            colors = {
                'Metal': 'red',
                'Semiconductor': 'green',
                'Insulator': 'blue'
            }

            value_counts = series.value_counts().sort_values(ascending=False)

            if top_n is not None:
                value_counts = value_counts.nlargest(top_n)

            value_counts = value_counts[::-1]

            fig, ax = plt.subplots()

            bars = ax.barh(value_counts.index, value_counts.values, linewidth=1.5, color=[colors.get(x, 'orange') for x in value_counts.index])

            for i, v in enumerate(value_counts.values):
                ax.text(v + 0.1, i, str(v), color='black', va='center')

            plt.xlabel('Counts', fontweight='bold')
            plt.ylabel(column_name, fontweight='bold')
            plt.title(f'Value Counts Bar Plot of {column_name}', fontweight='bold')

            ax.set_xlim(0, max(value_counts.values) * 1.1)

            st.pyplot(fig)

        plot_value_counts_material_category(resis['category'], 'category' )

    st.markdown(" ---")

resis_data = resis[resis["category"].notnull()]
sample_id = resis_data["sampleid"].unique()
filter_df = filter_df[filter_df["sampleid"].isin(sample_id)]
filter_df = filter_df.merge(resis_data[["sampleid", "category"]], on="sampleid", how="left")




with information :
    st.write(f"#### Materials that can be categorized")

    category = list(filter_df["category"].unique())
    # category.append("All")

    selected_category = st.multiselect("Select material category", category, default=category)
    if selected_category != "All":
        filter_df = filter_df[filter_df["category"].isin(selected_category)]
    else:
        filter_df = filter_df

    number_of_data = filter_df.shape[0]
    number_of_sample = filter_df["sampleid"].nunique()


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

    def calculate_pivot(data, **kwargs):
        """
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
        
        #pivot_final[f"result"] = pivot_final[properties[1]]**2/pivot_final[properties[0]]

        return pivot_final
    
   

    pivot_final = calculate_pivot(filter_df, **{f"property_{i}": prop for i, prop in enumerate(properties_filter)})
    # pivot_final = calculate_pivot(df, properties_filter)
    
    st.markdown("#### Pivot table material properties at 300K")
    st.markdown(f"Total data or sample plot : {pivot_final.shape[0]}")

    page_size = 5

    if 'start_idx' not in st.session_state:
        st.session_state.start_idx = 0  # Initialize start index

    format_mapping = {
        # 'Seebeck coefficient': '{:.6f}',  
        'Electrical resistivity': '{:.2e}',  
    }

    if properties_filter:
        format_column = properties_filter

    for column, format_str in format_mapping.items():
        if column in properties_filter:
            pivot_final[column] = pivot_final[column].apply(lambda x: format_str.format(x))
    st.dataframe(pivot_final.iloc[st.session_state.start_idx:st.session_state.start_idx + page_size])

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Previous'):
            if st.session_state.start_idx > 0:
                st.session_state.start_idx -= page_size
    with col2:
        if st.button('Next'):
            if st.session_state.start_idx + page_size < len(df):
                st.session_state.start_idx += page_size

    placeholder = st.empty()

    st.markdown(" ---")

if len(properties_filter) > 1 and "ZT" in properties_filter:

    with scatterplot :


        def scatterplot_thermal_pf(data, x_feature="Thermal conductivity", y_feature="Power factor", on=False, hue = None):
            
            num_data = data.shape[0]  

            plt.figure(figsize=(10, 6))

            sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=hue, palette='bright', alpha=0.7, s=50)

            top5_materials = data.sort_values(by="ZT", ascending=False).head(5)
            if on :
                offset = (5, 5)  
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

            colormap = plt.cm.magma

            norm = mcolors.Normalize(vmin=data['ZT'].min(), vmax=data['ZT'].max())

            scatter = plt.scatter(data[x_feature], data[y_feature], c=data['ZT'], cmap=colormap, norm=norm, alpha=0.7, s=50)

            if on:
                offset = (5, 5)  
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
      

            cbar = plt.colorbar(scatter)
            cbar.set_label("ZT")

            plt.tight_layout()
            st.pyplot(plt)
                       

        def scatterplot_thermal_pf_heatmap_with_contour(data: pd.DataFrame, 
                                                        x_feature: str = "Thermal conductivity", 
                                                        y_feature: str = "Power factor", 
                                                        on: bool = True) -> None:

            plt.figure(figsize=(10, 6))

            top5_materials = data.sort_values(by="ZT", ascending=False).head(5)

            colormap = plt.cm.magma
            norm = mcolors.Normalize(vmin=data['ZT'].min(), vmax=data['ZT'].max())

            x = pd.to_numeric(data[x_feature], errors='coerce')
            y = pd.to_numeric(data[y_feature], errors='coerce')

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            nbins = 100
            k = kde.gaussian_kde([x,y])
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            contour = plt.contourf(xi, yi, zi.reshape(xi.shape), cmap="viridis", alpha=0.7)

            scatter = plt.scatter(x, y, c=data.loc[mask, 'ZT'], cmap=colormap, norm=norm, alpha=0.7, s=20)

            if on:
                offset = (5, 5)
                for _, row in top5_materials.iterrows():
                    plt.annotate(
                        f"{row['composition']} ({row['materialfamily']})", 
                        (row[x_feature], row[y_feature]), 
                        textcoords="offset points", 
                        xytext=offset
                    )

            unit_x = df[df["propertyname_y"]==x_feature]["unitname_y"].value_counts().index[0]
            unit_y = df[df["propertyname_y"]==y_feature]["unitname_y"].value_counts().index[0]

            plt.xlabel(f"{x_feature} ({unit_x})")
            plt.ylabel(f"{y_feature} ({unit_y})")

            plt.title(f"{x_feature} vs {y_feature}")
            cbar = plt.colorbar(contour, format='')
            cbar.set_label("Density Data")

            plt.tight_layout()
            st.pyplot(plt) 

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

            plt.figure(figsize=(10, 6))

            colormap = plt.cm.magma
            norm = mcolors.Normalize(vmin=data['ZT'].min(), vmax=data['ZT'].max())

            x = pd.to_numeric(data[x_feature], errors='coerce')
            y = pd.to_numeric(data[y_feature], errors='coerce')
            z = pd.to_numeric(data[z_feature], errors='coerce')

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x = x[mask]
            y = y[mask]
            z = z[mask]

            if x.size == 0 or y.size == 0 or z.size == 0:
                raise ValueError("After cleaning the data, no valid points remain.")

            xi = np.linspace(x.min(), x.max(), 20)
            yi = np.linspace(y.min(), y.max(), 20)
            xi, yi = np.meshgrid(xi, yi)

            zi = griddata((x, y), z, (xi, yi), method='linear')

            contour_lines = plt.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
            contour_filled = plt.contourf(xi, yi, zi, levels=14, cmap=colormap)

            scatter = plt.scatter(x, y, c=z, cmap=colormap, s=30)

            cbar = plt.colorbar(contour_filled)
            cbar.set_label(z_feature)

            unit_x = df[df["propertyname_y"]==x_feature]["unitname_y"].value_counts().index[0]
            unit_y = df[df["propertyname_y"]==y_feature]["unitname_y"].value_counts().index[0]

            plt.xlabel(f"{x_feature} ({unit_x})")
            plt.ylabel(f"{y_feature} ({unit_y})")

            plt.title(f'Scatter plot with contour lines for {x_feature} vs {y_feature}')

            st.pyplot(plt)


        # if area_color:

        scatterplot_thermal_pf_heatmap_with_contour(pivot_final, x_feature=properties_filter[0], y_feature=properties_filter[1], on=on)
        
        st.markdown(" ---")



if len(properties_filter)>0:

    with st.expander(f"##### Show boxplot of selected properties"):

        # st.markdown(f" Boxplot of {properties_filter[0]} and {properties_filter[1]}")

        fig, ax = plt.subplots(nrows=len(properties_filter), ncols=1, figsize=(8, 6))

        colors = ['red', 'green', 'blue', 'orange', 'purple']

        for i, pro in enumerate(properties_filter):
            sns.boxplot(x=pro, data=pivot_final, ax=ax[i], palette=[colors[i]])
            plt.tight_layout()
        
        st.pyplot(fig)      

        st.markdown(" ---")

### Temperature Dependence ###




    
if len(properties_filter) > 0:

    with st.expander(f"##### Temperature dependence"):

        def data_temp(df=filter_df, propertyname="Seebeck coefficient", min_temp=300, max_temp=1000,hue="ZT", min_data = -10e50, max_data = 10e50):
             
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

            # def get_iterable(value):
            #     if isinstance(value, (list, tuple)):
            #         return value
            #     elif isinstance(value, dict):
            #         return [value]
            #     else:
            #         return [value]

            import ast

            def get_iterable(value):
                if isinstance(value, str):
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):                        
                        value = []

                if isinstance(value, (list, tuple)):
                    return value
                elif isinstance(value, dict):
                    return [value]
                else:
                    return [value]


            for _, row in pivot_final.iterrows():
                
                prop = get_iterable(row[property_name])
                zt_values = get_iterable(row['ZT'])
                
                for point_s, point_z in zip(prop, zt_values):
                    
                    # print(f"Type of point_s: {type(point_s)}, Value: {point_s}")  
                    # print(f"Type of point_z: {type(point_z)}, Value: {point_z}")  
                    
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

            plt.title(f"{property} vs Temperature")



            plt.tight_layout()
            st.pyplot(fig)
            # dataset = pd.read_csv('/mnt/data/dataset.csv')
            # dataset['Seebeck coefficient'] = dataset['Seebeck coefficient'].apply(ast.literal_eval)
            # dataset['ZT'] = dataset['ZT'].apply(ast.literal_eval)



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
            

