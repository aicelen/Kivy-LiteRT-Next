# Kivy-LiteRT-Next
Running LiteRT-Next on Android using Python, Pyjnius and Kivy.

Be careful when using this since lite-rt:2.0.x-alpha has two vulnerabilities:

https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2023-2976

https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-8908

LiteRT-Next provides easy state-of-the-art GPU acceleration granting huge performance uplifts compared to LiteRT.
This is the first example of using GPU acceleration for a .tflite model in Kivy. Compared to CPU (on LiteRT) it is around 2 times faster.

Your model need to be either unquantized or in FP16. Dynamic-Range Quantization is not working. 

Currently CPU Acceleration is a bit buggy but GPU Acceleration should be faster anyways.
