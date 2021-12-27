<?php
    $host = "localhost";
    $user = "root";
    $pass = "";
    $data = "site02";

    $mysqli = new mysqli($host, $user, $pass, $data);
	if($mysqli->connect_error) {
		echo "ERRO DE CONEXÃO COM O BANCO DE DADOS";
		exit();
	}  
?>