document.getElementById('predictionForm').addEventListener('submit', function(event) {
    const inputs = document.querySelectorAll('input[type="number"]');
    let valid = true;
    
    inputs.forEach(input => {
        const value = input.value.trim();
        if (value === '' || isNaN(value) || value < 0) {
            valid = false;
            input.classList.add('border-red-500');
        } else {
            input.classList.remove('border-red-500');
        }
    });
    
    if (!valid) {
        event.preventDefault();
        alert('Please ensure all inputs are valid non-negative numbers.');
    }
});