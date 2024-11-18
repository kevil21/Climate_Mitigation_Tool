const apk = "bad33d4d518cdc5169e27cd7f5c0d803";
let theday = parseInt(dayjs().format("DD"));
let currentDate = dayjs().format("DD/MM/YY");
let temperatureData = "";
let windData = "";
let humidityData = "";
let forecastData = $("#forecast");
let searchInput = $("#searchInput");
let searchButton = $("#searchButton");
let today = $("#today");
let inputGroupAppend = $("#input-group-append ");
let savedHistorial = $("#historial");
let searchHistory = [];
let setPlace;


// get the history of cities searched
const getHistorial = () => {
  const theHistorial = localStorage.getItem("savedSearchHistorial");
  return theHistorial ? JSON.parse(theHistorial) : [];
};

const setHistorial = () => {
  // duplicates removed
    let sh = removeDuplicates(searchHistory); 

    // Store and slice array to the specified limit(this case 5)
    localStorage.setItem("savedSearchHistorial", JSON.stringify(sh.slice(0, 5))); 
};


// remove duplicated items from the search history
const removeDuplicates = (arr) => {
  return arr.filter((item,
      index) => arr.indexOf(item) === index);
}

// retrieve the places from the local storage
const getSavedPlace = () => {
    const savedPlace = JSON.parse(localStorage.getItem("place"));
    
    // default to 'Surat' if no place is found
    return savedPlace ? savedPlace : "Surat"; 
};

// set searched places individually to the localStorage
const setSavedPlace = () => {
    if (searchInput.val().trim() !== "") {
        setPlace = searchInput.val();
        localStorage.setItem("place", JSON.stringify(setPlace))
        searchHistory.unshift(setPlace);
        setHistorial();

      }
};

// render history buttons
const renderHistoryButtons = () => {
    savedHistorial.empty();


    for (let i = 0; i < searchHistory.length && i < 5; i++) {
      //Create the buttons and add the history list array as values
      const historyBtn = $("<button class='btn btn-secondary mb-2'>").text(searchHistory[i].toUpperCase());

      //On clicking, trigger the cities found in the history
      historyBtn.on("click", function () {
        
        //adding them as parameter so that user can search those already saved, straight from the buttons
        findCityWeather(searchHistory[i]);

      });

      // Rendering the buttons to the history section
      savedHistorial.append(historyBtn);
    }

};

const getPlace = getSavedPlace();


//Function to find the cities the user wants to check
const findCityWeather = (getPlace) => {
    getPlace = getPlace;
  // query the api
  const theQuery = `https://api.openweathermap.org/data/2.5/forecast?q=${getPlace}&appid=${apk}`;

  fetch(theQuery)
    .then((response) => response.json())
    .then((data) => {

      // Check for data in the api
    if (data.list && data.list.length > 0) {
      temperatureData = Math.round(data.list[0].main.temp - 273.15);
      windData = data.list[0].wind.speed;
      humidityData = data.list[0].main.humidity;

      // Rendering data to the Current day weather section
      $("#today h3").text(`${getPlace} - ${currentDate}`);
      $("#today div img").attr(
        "src",
        ` https://openweathermap.org/img/wn/${data.list[1].weather[0].icon}.png` //Adding icon according to weather
      );
      $("#today p:eq(0)").text(`Temperature: ${temperatureData}°C`);
      $("#today p:eq(1)").text(`Wind: ${windData} KPH`);
      $("#today p:eq(2)").text(`Humidity: ${humidityData}%`);

      // Clear previous forecast data
      forecastData.empty(); 

        // Iterate for the five days  ahead
      for (let i = 1; i < 6; i++) {
        temperatureData = Math.round(data.list[i].main.temp - 273.15);
        windData = data.list[i].wind.speed;
        humidityData = data.list[i].main.humidity;

        // Add 5 days to the current date
        const currentDate = dayjs();
        const futureDate = currentDate.add(i, "day");
        const icon = `https://openweathermap.org/img/wn/${data.list[i].weather[0].icon}.png`;

        // Display the 5 days forecast
        const theFiveDays = `
                
                <div class="col five-day m-2 py-3 px-4">
                 <h5>${futureDate.format("DD/MM/YYYY")}</h5>
                 <div><img src="${icon}" width="60" ></div>
                 <p>Temperature: ${temperatureData}°C</p>
                 <p>Wind: ${windData} KPH</p>
                 <p>Humidity: ${humidityData}%</p>
                </div>
    
                `;

      //Render the forecast
       forecastData.append(theFiveDays);

      }
        // Update the saved place
        setSavedPlace(getPlace);
        
      }
    })// check for errors
    .catch((error) => {
        console.error("Error fetching data:", error);
      });
};

// Load search history on page load
$(document).ready(function () {
    searchHistory = getHistorial();
    renderHistoryButtons();
  });
 

// Save the place to localStorage just before the page is unloaded
window.addEventListener("beforeunload", function () {
    setSavedPlace(getSavedPlace());
  });

  // Initial call to findCityWeather
  findCityWeather(getSavedPlace());  