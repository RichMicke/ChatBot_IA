document.getElementById("chat-form").addEventListener("submit", async function(e) {
    e.preventDefault();
    const userInput = document.getElementById("user-input").value;
    document.getElementById("user-input").value = "";

    // Añade el mensaje del usuario al chat
    const chatbox = document.getElementById("chatbox");
    chatbox.innerHTML += `<div class="user-message">${userInput}</div>`;

    // Indicador de "escribiendo..."
    const typingIndicator = document.createElement("div");
    typingIndicator.classList.add("bot-message");
    typingIndicator.textContent = "Bot está escribiendo...";
    chatbox.appendChild(typingIndicator);

    // Hacer scroll hasta el final
    chatbox.scrollTop = chatbox.scrollHeight;

    // Llama al backend para obtener la respuesta del chatbot
    const response = await fetch("/get_response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    }).then(res => res.json());

    // Remueve el indicador de "escribiendo..."
    typingIndicator.remove();

    // Añade la respuesta del chatbot al chat
    chatbox.innerHTML += `<div class="bot-message">${response.response}</div>`;

    // Hacer scroll hasta el final
    chatbox.scrollTop = chatbox.scrollHeight;
});
