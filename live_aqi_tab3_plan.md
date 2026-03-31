# Live AQI Update for Tab3 - Implementation Plan

## Information Gathered:
- OpenWeather Air Pollution API confirmed working (`/data/2.5/air_pollution?lat&lon&appid`)
- Returns: aqi (1-5 scale), pm2_5, pm10, no2, so2, co, o3, nh3
- Same API_KEY as weather (e2fb24c6acbcce74b3b8679a6f5b35c0)
- Current Tab3 shows historical metrics - add live section before it

## Plan:
1. **New function**: `get_live_aqi(lat, lon)` - fetch pollution data, map aqi 1-5 to 0-500 scale
2. **New fragment**: `@st.fragment(run_every=300)` render_live_aqi() - metrics like weather tab
3. **Tab3 layout**: "🔴 LIVE AQI" → Historical metrics
4. **Fallback**: Show historical if API fails
5. **AQI scale**: 1=Good(0), 2=Moderate(50), 3=Unhealthy(100), 4=Very(200), 5=Hazardous(300)

## Code Changes (app.py):
```
# Add after get_live_weather()
@st.cache_data(ttl=300)
def get_live_aqi(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and 'list' in data:
            components = data['list'][0]['components']
            aqi_code = data['list'][0]['main']['aqi']
            # Map 1-5 to standard 0-500
            aqi_map = {1:25, 2:75, 3:125, 4:200, 5:300}
            return {
                'aqi': aqi_map.get(aqi_code, 100),
                'pm25': round(components.get('pm2_5', 0), 1),
                'pm10': round(components.get('pm10', 0), 1),
                'no2': round(components.get('no2', 0), 1),
                'so2': round(components.get('so2', 0), 1),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
    except:
        return None

@st.fragment(run_every=300)
def render_live_aqi(lat, lon, location):
    aqi_data = get_live_aqi(lat, lon)
    if aqi_data:
        cols = st.columns(5)
        with cols[0]: st.metric("📊 Live AQI", f"{aqi_data['aqi']}")
        with cols[1]: st.metric("🌫️ PM2.5", f"{aqi_data['pm25']} µg/m³")
        with cols[2]: st.metric("PM10", f"{aqi_data['pm10']} µg/m³")
        with cols[3]: st.metric("NO2", f"{aqi_data['no2']} µg/m³")
        with cols[4]: st.metric("SO2", f"{aqi_data['so2']} µg/m³")
        st.caption(f"Live: {location} | Updated: {aqi_data['timestamp']} (5min)")
    else:
        st.warning("🔌 Live AQI unavailable")

# In Tab3, after lat/lon detection:
if lat and lon:
    st.subheader("🔴 LIVE AQI Update")
    render_live_aqi(lat, lon, coord_location)
```

## Dependent Files:
- **app.py** (only)

## Follow-up:
Confirm lat/lon available in Tab3 → Add live section → Test API → attempt_completion

**Ready to implement?**

