css = '''
<style> 
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #f8f9fa;
}
.chat-message.user {
    background-color: #e9ecef;
    border-left: 5px solid #1565c0;  /* Azul institucional */
}
.chat-message.bot {
    background-color: #ffebee;  /* Fondo rojo claro */
    border-left: 5px solid #c62828;  /* Borde rojo oscuro */
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 68px;
    max-height: 68px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #212529;
}
'''

logo = '''
<div style="margin-bottom: 15px; text-align: center;">
    <img src="https://seeklogo.com/images/A/alcaldia-mayor-de-bogota-logo-CA468F866B-seeklogo.com.png" 
         alt="Logo Alcaldia BOgota" style="max-width: 30%; height: auto;">
    <img src="https://public.tableau.com/avatar/fd94acf3-58a6-4811-b3fa-94c2662ce866.jpeg" 
        alt="Logo GOVLAB" style="max-width: 25%; height: auto;">
</div>
'''