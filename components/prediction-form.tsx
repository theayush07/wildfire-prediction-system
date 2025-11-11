"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface PredictionFormProps {
  onPredictionComplete: (result: {
    risk: number
    riskLevel: "Low" | "Medium" | "High" | "Critical"
    factors: string[]
  }) => void
}

export function PredictionForm({ onPredictionComplete }: PredictionFormProps) {
  const [temperature, setTemperature] = useState([25])
  const [humidity, setHumidity] = useState([50])
  const [windSpeed, setWindSpeed] = useState([15])
  const [rainfall, setRainfall] = useState([2])
  const [month, setMonth] = useState("jul")
  const [day, setDay] = useState("sun")
  const [loading, setLoading] = useState(false)

  const handlePredict = async () => {
    setLoading(true)

    // Simulate prediction logic
    await new Promise((resolve) => setTimeout(resolve, 1000))

    // Simple rule-based prediction for demo
    let riskScore = 0
    const factors: string[] = []

    // Temperature contribution
    if (temperature[0] > 30) {
      riskScore += 0.3
      factors.push("High temperature")
    } else if (temperature[0] > 25) {
      riskScore += 0.15
    }

    // Humidity contribution (inverse)
    if (humidity[0] < 30) {
      riskScore += 0.25
      factors.push("Low humidity")
    } else if (humidity[0] < 50) {
      riskScore += 0.1
    }

    // Wind speed contribution
    if (windSpeed[0] > 25) {
      riskScore += 0.2
      factors.push("Strong winds")
    } else if (windSpeed[0] > 15) {
      riskScore += 0.1
    }

    // Rainfall contribution (inverse)
    if (rainfall[0] < 1) {
      riskScore += 0.15
      factors.push("Low recent rainfall")
    }

    // Season contribution
    if (["jul", "aug", "sep"].includes(month)) {
      riskScore += 0.1
      factors.push("High-risk season")
    }

    riskScore = Math.min(riskScore, 1)

    let riskLevel: "Low" | "Medium" | "High" | "Critical"
    if (riskScore < 0.25) riskLevel = "Low"
    else if (riskScore < 0.5) riskLevel = "Medium"
    else if (riskScore < 0.75) riskLevel = "High"
    else riskLevel = "Critical"

    if (factors.length === 0) {
      factors.push("Favorable conditions")
    }

    onPredictionComplete({
      risk: riskScore,
      riskLevel,
      factors,
    })

    setLoading(false)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Environmental Conditions</CardTitle>
        <CardDescription>Adjust parameters to predict wildfire risk</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-6 sm:grid-cols-2">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="temperature">Temperature (°C)</Label>
              <span className="text-sm font-medium">{temperature[0]}°C</span>
            </div>
            <Slider id="temperature" min={0} max={45} step={1} value={temperature} onValueChange={setTemperature} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="humidity">Relative Humidity (%)</Label>
              <span className="text-sm font-medium">{humidity[0]}%</span>
            </div>
            <Slider id="humidity" min={0} max={100} step={1} value={humidity} onValueChange={setHumidity} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="wind">Wind Speed (km/h)</Label>
              <span className="text-sm font-medium">{windSpeed[0]} km/h</span>
            </div>
            <Slider id="wind" min={0} max={50} step={1} value={windSpeed} onValueChange={setWindSpeed} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="rainfall">Rainfall (mm)</Label>
              <span className="text-sm font-medium">{rainfall[0]} mm</span>
            </div>
            <Slider id="rainfall" min={0} max={10} step={0.1} value={rainfall} onValueChange={setRainfall} />
          </div>

          <div className="space-y-3">
            <Label htmlFor="month">Month</Label>
            <Select value={month} onValueChange={setMonth}>
              <SelectTrigger id="month">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="jan">January</SelectItem>
                <SelectItem value="feb">February</SelectItem>
                <SelectItem value="mar">March</SelectItem>
                <SelectItem value="apr">April</SelectItem>
                <SelectItem value="may">May</SelectItem>
                <SelectItem value="jun">June</SelectItem>
                <SelectItem value="jul">July</SelectItem>
                <SelectItem value="aug">August</SelectItem>
                <SelectItem value="sep">September</SelectItem>
                <SelectItem value="oct">October</SelectItem>
                <SelectItem value="nov">November</SelectItem>
                <SelectItem value="dec">December</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label htmlFor="day">Day of Week</Label>
            <Select value={day} onValueChange={setDay}>
              <SelectTrigger id="day">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mon">Monday</SelectItem>
                <SelectItem value="tue">Tuesday</SelectItem>
                <SelectItem value="wed">Wednesday</SelectItem>
                <SelectItem value="thu">Thursday</SelectItem>
                <SelectItem value="fri">Friday</SelectItem>
                <SelectItem value="sat">Saturday</SelectItem>
                <SelectItem value="sun">Sunday</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <Button onClick={handlePredict} disabled={loading} className="w-full" size="lg">
          {loading ? "Calculating Risk..." : "Predict Fire Risk"}
        </Button>
      </CardContent>
    </Card>
  )
}
