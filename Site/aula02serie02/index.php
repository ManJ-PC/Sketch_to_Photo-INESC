<?php
    require("Configs/connection.php");
    session_start();
?>

<!DOCTYPE html>
<html>
	<head>
        <meta charset="UTF-8">
        <title> Nome do Site </title>
        <link rel="stylesheet" type="text/css" href="Styles/index.css" />
        <script type="text/javascript">
            function jsImagem() {
                document.getElementById("fileImagem").click;
                return false;
            }
        </script>
    </head>
    <body>
        <div id="pContainer">
            <div id="pMenu">
                <ul>
                    <li> <a href="#"> Home </a> </li>
                    <li> <a href="#"> Sobre </a> </li>
                    <li> <a href="#"> Contato </a> </li>
                    <li> <a href="#"> Imagens </a> </li>
                </ul>
            </div>
            <div id="pTopo">
                <img src="Images/logo.png" alt="Logo do site Nome do Site" title="Logo do site" />
            </div>
            <div id="pConteudo">  <br /> <br />
                <div id="pSobre">
                    <strong> Nome do Site </strong> é um sistema de carregamento de imagens gratuito, que você pode compartilhar suas imagens com todas as pessoas que você quiser, através de um link que é gerado após você carregar sua imagem.
                </div>
				<form action="" method="POST" enctype="multipart/form-data">
                    <div id="pUpload">
                        <a OnClick="jsImagem()" id="pEscolher"> Escolher Imagem </a>
                        <input type="file" id="fileImagem" name="fileImagem" />
                    </div>
                    <input id="pButton" name="pButton" type="submit" />
                </form>
                <div id="pLinkk">
                    <?php
                        if(isset($_POST["pButton"])) {
                            $arquivo = $_FILES["fileImagem"]["name"];
                            $tipo    = $_FILES["fileImagem"]["type"];
                            $tamanho = $_FILES["fileImagem"]["size"];
                            $tmp     = $_FILES["fileImagem"]["tmp_name"];
                            $erro    = $_FILES["fileImagem"]["error"];
                            if($erro == 0) {
                                echo "<span> Link: </span>";
                                $urll = $_SESSION["pURL"];
                                echo "<input type='text' id='pLink' value='$urll' />";
                                unset($_SESSION["pURL"]);
                            } else {
                                echo "<script> alert('ERRO AO CARREGAR IMAGEM'); location.href='index.php' </script>";
                            }
                        }
                    ?>
                </div>
            </div>
    
            <div id="pRodape">
                <ul>
                    <li> <a href="#"> Nome do Site </a> &copy 2014 - Todos os direitos reservados </li>
                    <li> <a href="#"> Sobre Nós </a> </li>
                    <li> <a href="#"> Contato </a> </li>
                </ul>
            </div>
        </div>
    </body>
</html>

<?php
    if(isset($_POST["pButton"])) {
        $arquivoTMP = $_FILES["fileImagem"]["tmp_name"];
        $nomeFile   = $_FILES["fileImagem"]["name"];
        
        $pExtensao = strrchr($nomeFile, '.');
        $pExtensao = strtolower($pExtensao);
        $pNovoNome = rand(0, 99999999999) . $pExtensao;
        $pDestino  = "uploads/" . $pNovoNome;
        
        if(move_uploaded_file($arquivoTMP, $pDestino)) {
            $link = "http://localhost/serie02/uploads/" . $pNovoNome;
            $_SESSION["pURL"] = $link;
            $insert = $mysqli->query("INSERT INTO `imagens`(`NomeTRUE`, `NomeFALSE`, `Link`) VALUES ('$nomeFile', '$pNovoNome', '$link')");
            if($insert) {
                echo "<script> alert('IMAGEM CARREGADA COM SUCESSO!'); </script>";      
            } else {
                echo $mysqli->error;   
            }
        } else {
             echo "<script> alert('ERRO AO CARREGAR IMAGEM'); location.href='index.php' </script>";  
        }
    }
?><?php
    if(isset($_POST["pButton"])) {
        $arquivoTMP = $_FILES["fileImagem"]["tmp_name"];
        $nomeFile   = $_FILES["fileImagem"]["name"];
        
        $pExtensao = strrchr($nomeFile, '.');
        $pExtensao = strtolower($pExtensao);
        $pNovoNome = rand(0, 99999999999) . $pExtensao;
        $pDestino  = "uploads/" . $pNovoNome;
        
        if(move_uploaded_file($arquivoTMP, $pDestino)) {
            $link = "http://localhost/serie02/uploads/" . $pNovoNome;
            $_SESSION["pURL"] = $link;
            $insert = $mysqli->query("INSERT INTO `imagens`(`NomeTRUE`, `NomeFALSE`, `Link`) VALUES ('$nomeFile', '$pNovoNome', '$link')");
            if($insert) {
                echo "<script> alert('IMAGEM CARREGADA COM SUCESSO!'); </script>";      
            } else {
                echo $mysqli->error;   
            }
        } else {
             echo "<script> alert('ERRO AO CARREGAR IMAGEM'); location.href='index.php' </script>";  
        }
    }
?>