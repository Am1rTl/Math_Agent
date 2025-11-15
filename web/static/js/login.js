const loginForm = document.getElementById('login-form');
const errorMessage = document.getElementById('error-message');

loginForm.addEventListener('submit', async (e) => {
	e.preventDefault();

	const username = document.getElementById('username').value;
	const password = document.getElementById('password').value;

	try {
		const response = await fetch('/api/login', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ username, password }),
		});

		if (response.ok) {
			const data = await response.json();
			localStorage.setItem('token', data.token);
			window.location.href = '/';
		} else {
			const error = await response.json();
			errorMessage.textContent = error.error || 'Ошибка входа';
		}
	} catch (error) {
		errorMessage.textContent = 'Ошибка сети';
	}
});
