# Import required libraries
import streamlit as st
import json
import clipboard

from main import genetic_algorithm, polish_errors, calculate_errors, update_genetic_algorithm, calculate_diff

# Initialize session state
if 'services' not in st.session_state:
    st.session_state.services = {}
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'solution' not in st.session_state:
    st.session_state.solution = {}

# App title
st.title('Services and Users Assignment Center')

# Add sliders for population_size, num_generations, and mutation_rate
st.subheader('Genetic Algorithm Parameters')
population_size = st.slider('Population Size', min_value=500, max_value=5000, value=1500, step=100)
num_generations = st.slider('Number of Generations', min_value=1000, max_value=10000, value=2500, step=250)
mutation_rate = st.slider('Mutation Rate', min_value=0.0, max_value=1.0, value=0.01, step=0.05)

# Button to run the genetic algorithm
new_generation_run = st.button('Run new solution')
update_generation_run = st.button('Update solution')

if new_generation_run:
    # Call the genetic_algorithm function and get the best_solution
    best_solution = genetic_algorithm(
        services=st.session_state.services,
        users=st.session_state.users,
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate
    )

    # Save the state of the current best solution
    st.session_state.solution = best_solution

    # Convert the best_solution to JSON
    best_solution_json = json.dumps(st.session_state.solution, indent=4)
    best_solution_errors = calculate_errors(st.session_state.solution, st.session_state.services,
                                            st.session_state.users)
    best_solution_errors = polish_errors(best_solution_errors)
    best_solution_errors_json = json.dumps(best_solution_errors, indent=4)

    # Display the output JSON in a read-only form
    st.subheader('Best solution JSON')
    st.text_area('Best solution',
                 value=best_solution_json, height=400, max_chars=None, key=None, disabled=True)
    st.text_area('Unmet constraints',
                 value=best_solution_errors_json, height=200, max_chars=None, key=None, disabled=True)

    if st.button('Copy solution to Clipboard'):
        clipboard.copy(best_solution_json)
    if st.button('Copy unmet constraints to Clipboard'):
        clipboard.copy(best_solution_errors_json)

if update_generation_run:
    # Call the genetic_algorithm function and get the best_solution
    best_updated_solution = update_genetic_algorithm(
        prev_solution=st.session_state.solution,
        updated_services=st.session_state.services,
        updated_users=st.session_state.users,
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate
    )

    change_report = calculate_diff(best_updated_solution, st.session_state.solution)
    change_report_json = json.dumps(change_report, indent=4)

    # Convert the best_solution to JSON
    best_updated_solution_json = json.dumps(st.session_state.solution, indent=4)
    best_updated_solution_errors = calculate_errors(
        st.session_state.solution, st.session_state.services, st.session_state.users
    )
    best_updated_solution_errors = polish_errors(best_updated_solution_errors)
    best_updated_solution_errors_json = json.dumps(best_updated_solution_errors, indent=4)

    # Display the output JSON in a read-only form
    st.subheader('Best updated solution JSON')
    st.text_area('Best updated solution',
                 value=best_updated_solution_json, height=600, max_chars=None, key=None, disabled=True)
    st.text_area('Updated unmet constraints',
                 value=best_updated_solution_errors_json, height=300, max_chars=None, key=None, disabled=True)
    st.text_area('Change report',
                 value=change_report_json, height=200, max_chars=None, key=None, disabled=True)

    if st.button('Copy updated solution to Clipboard'):
        clipboard.copy(best_updated_solution_json)
    if st.button('Copy updated unmet constraints to Clipboard'):
        clipboard.copy(best_updated_solution_errors_json)
    if st.button('Save this updated solution over the last fully generated one'):
        # Save the state of the current best solution
        st.session_state.solution = best_updated_solution

# Sidebar for uploading previously generated JSON
with st.sidebar.expander('Previously generated solution JSON'):
    uploaded_solution_json = st.text_area('Paste your previously generated JSON here',
                                          value=json.dumps(st.session_state.get('solution', ''), indent=4))
    merge_json = st.button('Upload previously generated JSON')
    reset_json = st.button('Reset previously generated JSON')

    if reset_json:
        st.session_state.solution = {}

    if merge_json and uploaded_solution_json:
        try:
            st.session_state.solution = json.loads(uploaded_solution_json)
            st.success('JSON loaded successfully')
        except json.JSONDecodeError:
            st.error('Invalid JSON format')

# Sidebar for uploading previously user and service description JSON
with st.sidebar.expander('Previously generated user and service description JSON'):
    previously_generated_user_service_json = {
        'services': st.session_state.services,
        'users': st.session_state.users
    }
    uploaded_user_service_json = st.text_area('Paste your user and service JSON here',
                                              value=json.dumps(previously_generated_user_service_json, indent=4)
                                              )
    merge_json = st.button('Merge user and service description JSON')
    reset_json = st.button('Reset user and service description JSON')

    if reset_json:
        st.session_state.services = {}
        st.session_state.users = {}

    if merge_json and uploaded_user_service_json:
        try:
            loaded_data = json.loads(uploaded_user_service_json)
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
        if service_key:
            if st.button('Load Service'):
                st.session_state.service_name = service_key
                st.session_state.min_val = st.session_state.services[service_key]['min']
                st.session_state.rec_val = st.session_state.services[service_key]['rec']
                st.session_state.max_val = st.session_state.services[service_key]['max']
                st.session_state.priority = st.session_state.services[service_key]['priority']
            if st.button('Drop Service'):
                del st.session_state.services[service_key]

    elif object_type == 'User':
        user_key = st.selectbox('Select a user', list(st.session_state.users.keys()), key='update_user_key')
        if user_key:
            if st.button('Load User'):
                st.session_state.user_name = user_key
                st.session_state.max_assignments = st.session_state.users[user_key]['max_assignments']
                st.session_state.preferences = st.session_state.users[user_key]['preferences']
                st.session_state.cannot_assign = st.session_state.users[user_key]['cannot_assign']
            if st.button('Drop User'):
                del st.session_state.users[user_key]

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
st.subheader('Generated user and services JSON')
st.code(json_data, language='json')

# Button to copy JSON to clipboard
if st.button('Copy JSON to Clipboard'):
    clipboard.copy(json_data)
    st.success('JSON copied to clipboard!')
