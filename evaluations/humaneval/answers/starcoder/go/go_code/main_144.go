package main 
 
import (          "strings" 
 )

func Simplify(x, n string) bool {
    x_num, x_den := strings.Split(x, "/")[0], strings.Split(x, "/")[1]
    n_num, n_den := strings.Split(n, "/")[0], strings.Split(n, "/")[1]
    x_num, x_den, n_num, n_den := int(x_num), int(x_den), int(n_num), int(n_den)
    return (x_num * n_den) == (x_den * n_num)
}