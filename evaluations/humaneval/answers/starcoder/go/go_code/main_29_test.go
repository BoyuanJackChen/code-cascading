package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestFilterByPrefix(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]string{}, FilterByPrefix([]string{}, "john"))
    assert.Equal([]string{"xxx", "xxxAAA", "xxx"}, FilterByPrefix([]string{"xxx", "asd", "xxy", "john doe", "xxxAAA", "xxx"}, "xxx"))
}
