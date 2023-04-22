# Import required libraries
import streamlit as st
import json
import clipboard

from main import genetic_algorithm, polish_errors, calculate_errors

# Initialize session state
if 'services' not in st.session_state:
    st.session_state.services = {}
if 'users' not in st.session_state:
    st.session_state.users = {}

# App title
st.title('Services and Users JSON Builder')

# Add sliders for population_size, num_generations, and mutation_rate
st.subheader('Genetic Algorithm Parameters')
population_size = st.slider('Population Size', min_value=500, max_value=5000, value=2500, step=100)
num_generations = st.slider('Number of Generations', min_value=1000, max_value=10000, value=5000, step=500)
mutation_rate = st.slider('Mutation Rate', min_value=0.0, max_value=1.0, value=0.01, step=0.01)

# Button to run the genetic algorithm
if st.button('Run Genetic Algorithm'):
    # Call the genetic_algorithm function and get the best_solution
    best_solution = genetic_algorithm(st.session_state.services, st.session_state.users, population_size,
                                      num_generations, mutation_rate)

    # Convert the best_solution to JSON
    best_solution_json = json.dumps(best_solution, indent=4)
    best_solution_errors = calculate_errors(best_solution, st.session_state.services, st.session_state.users)
    best_solution_errors = polish_errors(best_solution_errors)
    best_solution_errors = json.dumps(best_solution_errors, indent=4)

    # Display the output JSON in a read-only form
    st.subheader('Best Solution JSON')
    st.text_area('Best Solution', value=best_solution_json, height=400, max_chars=None, key=None, disabled=True)
    st.text_area('Unmet constraints', value=best_solution_errors, height=200, max_chars=None, key=None, disabled=True)

    if st.button('Copy solution to Clipboard'):
        clipboard.copy(best_solution_json)
        st.success('JSON copied to clipboard!')
    if st.button('Copy unmet constraints to Clipboard'):
        clipboard.copy(best_solution_errors)
        st.success('JSON copied to clipboard!')

# Sidebar for uploading previously generated JSON
with st.sidebar.expander('Upload previously generated JSON'):
    uploaded_json = st.text_area('Paste your JSON here')
    merge_json = st.button('Merge with JSON')
    reset_json = st.button('Reset JSON')

    if reset_json:
        st.session_state.services = {}
        st.session_state.users = {}

    if merge_json and uploaded_json:
        try:
            loaded_data = json.loads(uploaded_json)
            st.session_state.services.update(loaded_data.get('services', {}))
            st.session_state.users.update(loaded_data.get('users', {}))
            st.success('JSON loaded successfully')
        except json.JSONDecodeError:
            st.error('Invalid JSON format')

# Update existing user or service object
with st.sidebar.expander('Update existing user or service'):
    object_type = st.selectbox('Choose object type', ('Service', 'User'))

    if object_type == 'Service':
        service_key = st.selectbox('Select a service', list(st.session_state.services.keys()), key='update_service_key')
        if service_key and st.button('Load Service'):
            st.session_state.service_name = service_key
            st.session_state.min_val = st.session_state.services[service_key]['min']
            st.session_state.rec_val = st.session_state.services[service_key]['rec']
            st.session_state.max_val = st.session_state.services[service_key]['max']
            st.session_state.priority = st.session_state.services[service_key]['priority']

    elif object_type == 'User':
        user_key = st.selectbox('Select a user', list(st.session_state.users.keys()), key='update_user_key')
        if user_key and st.button('Load User'):
            st.session_state.user_name = user_key
            st.session_state.max_assignments = st.session_state.users[user_key]['max_assignments']
            st.session_state.preferences = st.session_state.users[user_key]['preferences']
            st.session_state.cannot_assign = st.session_state.users[user_key]['cannot_assign']

# Add a service form
with st.form(key='service_form'):
    st.subheader('Add a Service')
    service_name = st.text_input('Service Name', value=st.session_state.get('service_name', ''))
    min_val = st.number_input('Minimum Value', value=st.session_state.get('min_val', 0))
    rec_val = st.number_input('Recommended Value', value=st.session_state.get('rec_val', 0))
    max_val = st.number_input('Maximum Value', value=st.session_state.get('max_val', 0))
    priority = st.number_input('Priority', value=st.session_state.get('priority', 0))
    submit_service = st.form_submit_button('Save Service')

# Add a user form
with st.form(key='user_form'):
    st.subheader('Add a User')
    user_name = st.text_input('User Name', key='user_name', value=st.session_state.get('user_name', ''))
    max_assignments = st.number_input('Max Assignments', value=st.session_state.get('max_assignments', 0),
                                      key='max_assignments')
    preferences = st.multiselect('Preferences', options=list(st.session_state.services.keys()),
                                 default=st.session_state.get('preferences', []), key='preferences')
    cannot_assign = st.multiselect('Cannot Assign', options=list(st.session_state.services.keys()),
                                   default=st.session_state.get('cannot_assign', []), key='cannot_assign')
    submit_user = st.form_submit_button('Save User')

# Add the submitted service to the services dictionary
if submit_service:
    st.session_state.services[service_name] = {
        'min': min_val,
        'rec': rec_val,
        'max': max_val,
        'priority': priority
    }

# Add the submitted user to the users dictionary
if submit_user:
    st.session_state.users[user_name] = {
        'max_assignments': max_assignments,
        'preferences': preferences,
        'cannot_assign': cannot_assign
    }

# Combine services and users dictionaries
combined_data = {
    'services': st.session_state.services,
    'users': st.session_state.users
}

# Convert combined_data to JSON
json_data = json.dumps(combined_data, indent=4)

# Display the generated JSON
st.subheader('Generated JSON')
st.code(json_data, language='json')

# Button to copy JSON to clipboard
if st.button('Copy JSON to Clipboard'):
    clipboard.copy(json_data)
    st.success('JSON copied to clipboard!')
