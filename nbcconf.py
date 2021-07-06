c = get_config()

# Specify the conversion options
c.NbConvertApp.export_format = 'notebook'
c.NbConvertApp.use_output_suffix = False
c.FilesWriter.build_directory = ''

c.ExecutePreprocessor.enabled = True
c.ExecutePreprocessor.timeout = -1

# Specify which notebooks should be converted
notebooks = list()

#notebooks.append('notebooks/')

# Add the baseline estimation
#notebooks.append('notebooks/01.1 - Load data for seasonal baseline estimation.ipynb')
notebooks.append('notebooks/01.2 - Estimate average energy use baseline seasonal trend.ipynb')
notebooks.append('notebooks/01.3 - Compute baseline for each home.ipynb')

# Add data prepartion for energy estimates of the defined periods
#notebooks.append('notebooks/02.1 - Prepare the data for the energy estimation.ipynb')
#notebooks.append('notebooks/02.2 - Check data quality for sampling approach.ipynb')
#notebooks.append('notebooks/02.3 - Check number of available samples (lower bound of all homes).ipynb')

# Add the notebook to compute the energy estimates for each period per home
notebooks.append('notebooks/03 - Sample average energy consumption.ipynb')

# Add the notebook to compute the rolling average energy estimates per home
notebooks.append('notebooks/04 - Compute rolling average energy estimate.ipynb')

# Add the notebook to compute the gas estimates
#notebooks.append('notebooks/05 - Compute gas use estimates.ipynb')

c.NbConvertApp.notebooks = notebooks
