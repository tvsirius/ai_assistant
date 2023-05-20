// set up basic variables for app

const record = document.querySelector('.record');
const border = document.querySelector('.border');
const canvas = document.querySelector('.visualizer');
const mainSection = document.querySelector('.main-controls');
const printout = document.querySelector('.printout');
const canvasCtx = canvas.getContext("2d");

let recording = false;
let audioCtx;

if (navigator.mediaDevices.getUserMedia) {
  const constraints = { audio: true };
  // this stores the data for the audio blob
  let chunks = [];

  let onSuccess = function(stream) {
    const mediaRecorder = new MediaRecorder(stream);
    visualize(stream);
    // when the user clicks on the record button
    record.onclick = function() {
      // start the recording
      if (recording == false) {
        mediaRecorder.start();
        record.style.background = "red";
        record.innerHTML = "<i class='fa-solid fa-stop'></i>";
        recording = true;
      } else {
        // stop the recording
        mediaRecorder.stop();
        record.style.background = "";
        record.innerHTML = "<i class='fa-solid fa-microphone'></i>";
        recording = false;
      }
    }
    // when the user clicks on the stop button
mediaRecorder.onstop = function(e) {
      // create an audio blob in webm format
      const blob = new Blob(chunks, { type: "audio/webm" });
      // add it to the form
      const formData = new FormData();
      console.log('sending data')
      formData.append('audio', blob, 'recording.webm');
      // send the audio blob to the server
      fetch('/record', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
            if (!(data["input"]==='' && data["output"]==="")){
                console.log('html steps: 1, i got data,\ninput=', data["input"],'\noutput=', data["output"]);
                out='<b>Human: </b>'+ data["input"]+'<br><b>Ai: </b>'+data["output"]+'<br>'
                resultElement.innerHTML += out
              console.log('result updated');
        // move to the bottom of the page
        var inputField = document.getElementById('myInputField');
        inputField.scrollIntoView();
              console.log('scrolled');
// --- TEXT TO SPEECH
        let speech = new SpeechSynthesisUtterance();
                     console.log('speech step 1 done');
        speech.text = data["output"]
                     console.log('speech step 2 done');
        speech.voice = speechSynthesis.getVoices()[0];
                     console.log('speech step 3 done');
        speechSynthesis.speak(speech);
                     console.log('speech step 4 done');
        }
      })
      .then(console.log('client side  work done'))
      .catch((error) => {
        console.error('Error:', error);
      });

//      fetch('/', {
//      method: 'GET'
//      }).then(response => {
//      // Handle the response if needed
//      console.log('GET request to "/" completed successfully.');
////      location.reload();
////      console.log('location reload completed successfully');
//    })
//    .catch((error) => {
//      console.error('Error:', error);
//    });


      chunks = [];
    }
      //


    // add the data into chunks when its available
    mediaRecorder.ondataavailable = function(e) {
      chunks.push(e.data);
    }
  }
  let onError = function(err) {
    console.log('The following error occured: ' + err);
  }
  navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);
} 
else {
   console.log('getUserMedia not supported on your browser!');
}

// display the visualization
function visualize(stream) {
  if(!audioCtx) {
    audioCtx = new AudioContext();
  }

  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  source.connect(analyser);
  draw()

  function draw() {
    const WIDTH = canvas.width
    const HEIGHT = canvas.height;
    requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArray);
    canvasCtx.fillStyle = 'aliceblue';
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'LightSteelBlue';
    canvasCtx.beginPath();

    let sliceWidth = WIDTH * 1.0 / bufferLength;
    let x = 0;
    for(let i = 0; i < bufferLength; i++) {
      let v = dataArray[i] / 128.0;
      let y = v * HEIGHT/2;
      if(i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height/2);
    canvasCtx.stroke();
  }
}

window.onresize = function() {
  canvas.width = mainSection.offsetWidth;
}

window.onresize();