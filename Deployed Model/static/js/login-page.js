
 loginForm = document.getElementById("login-form");
  loginButton = document.getElementById("login-form-submit");
   loginErrorMsg = document.getElementById("login-error-msg");

loginButton.addEventListener("click", (e) => {
    e.preventDefault();
     username = loginForm.username.value;
     password = loginForm.password.value;

    if (username === "user" && password === "web_dev") {
        alert("You have successfully logged in.");
        window.location.assign("index.html");
    } else {
        loginErrorMsg.style.opacity = 1;
    }
})
