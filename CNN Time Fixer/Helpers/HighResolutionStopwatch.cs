using System;
using System.Runtime.InteropServices;

namespace CNN_Time_Fixer.Helpers
{
    public class HighResolutionStopwatch
    {
        [DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);

        [DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceFrequency(out long lpFrequency);

        private long _startTime;
        private long _stopTime;
        private readonly long _frequency;

        public HighResolutionStopwatch()
        {
            if (!QueryPerformanceFrequency(out _frequency))
            {
                throw new InvalidOperationException("High-resolution performance counter not supported.");
            }
        }

        public void Start()
        {
            QueryPerformanceCounter(out _startTime);
        }

        public void Stop()
        {
            QueryPerformanceCounter(out _stopTime);
        }

        public double ElapsedMilliseconds
        {
            get
            {
                long elapsed = _stopTime - _startTime;
                return (elapsed * 1000.0) / _frequency;
            }
        }

        public void Reset()
        {
            _startTime = 0;
            _stopTime = 0;
        }
    }
}