const API_URL = 'http://127.0.0.1:5000/'

// This will upload the file after having read it
function upload(file) {
  let formData = new FormData()
  formData.append('file', file)

  fetch(API_URL + 'pix2pix', {
    // Your POST endpoint
    method: 'POST',
    body: formData // This is your file object
  })
    .then(
      response => response.json() // if the response is a JSON object
    )
    .then(success => {
      console.log(success) // Handle the success response object

      let load = document.getElementById('loading')
      load.style.display = 'none'

      let divs = document.getElementById('calculated')
      divs.style.display = 'block'

      let skeImg = document.getElementById('generated')
      skeImg.src = success[0]

      let photosSim = document.getElementsByClassName('sim-img')
      console.log(photosSim)

      let output = document.getElementById('sketch')
      console.log(output.src)
      console.log(skeImg.src)
      for (let i = 0; i < success[1].length && i < photosSim.length; i++) {
        photosSim[i].src = success[1][i][0]
        if (photosSim[i].src.split('/').pop() === skeImg.src.split('/').pop()) {
          photosSim[i].classList.add('box-3')
        } else photosSim[i].classList.remove('box-3')
        console.log(photosSim[i].src.split('/').pop())
        console.log(skeImg.src.split('/').pop())
      }
    })
    .catch(
      error => console.log(error) // Handle the error response object
    )
}

// Event handler executed when a file is selected
function onSelectFile() {
  // Select your input type file and store it in a variable
  const input = document.getElementById('fileinput')

  let load = document.getElementById('loading')
  load.style.display = 'block'

  upload(input.files[0])
}

var loadFile = function(event) {
  let sketchDiv = document.getElementById('sketchDiv')
  sketchDiv.style.display = 'block'
  let output = document.getElementById('sketch')
  output.src = URL.createObjectURL(event.target.files[0])
  let divs = document.getElementById('calculated')
  divs.style.display = 'none'
}

/*
function jsImagem() {
  document.getElementById('fileImagem').click
  return false
}

function test() {
  console.log(45)
}
console.log('imported')

const onSelectFile = () => upload(input.files[0])


let uploadImg = function(event) {
  let url = 'http://127.0.0.1:5000/'
  fetch(url, {
    method: 'GET',
    mode: 'cors'
    //body: formData
  })
    .then(function(response) {
      // The response is a Response instance.
      // You parse the data into a useable format using `.json()`
      //return response.json()
      return response.text()
    })
    .then(function(data) {
      // `data` is the parsed version of the JSON returned from the above endpoint.
      console.log(data) // { "userId": 1, "id": 1, "title": "...", "body": "..." }
    })
}


*/
